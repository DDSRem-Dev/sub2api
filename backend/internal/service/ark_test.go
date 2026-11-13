//go:build unit

package service

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func arkTestAccount() *Account {
	return &Account{ID: 84, Platform: PlatformArk, Type: AccountTypeAPIKey, Credentials: map[string]any{"api_key": "test-ark-key", "api_protocol": APIProtocolAdaptive}}
}

func TestArkPlatformCatalogAndIsolation(t *testing.T) {
	ids := DefaultArkModelIDs()
	require.Len(t, ids, 54)
	seen := map[string]bool{}
	for _, id := range ids {
		require.False(t, seen[id], id)
		seen[id] = true
		platform, ok := DetectModelPlatform(id)
		require.True(t, ok, id)
		require.Equal(t, PlatformArk, platform, id)
	}
	ids[0] = "mutated"
	require.NotEqual(t, ids[0], DefaultArkModelIDs()[0])
	for model, want := range map[string]string{"deepseek-chat": PlatformDeepseek, "kimi-k2.5": PlatformKimi, "glm-5": PlatformZhipu, "ark/custom": PlatformArk} {
		got, ok := DetectModelPlatform(model)
		require.True(t, ok)
		require.Equal(t, want, got)
	}
	a := arkTestAccount()
	require.True(t, a.IsMultiProtocolAPIKey())
	require.False(t, IsCNProvider(PlatformArk))
	require.Equal(t, DefaultArkBaseURL, a.GetOpenAIBaseURL())
	require.Equal(t, DefaultArkAnthropicBaseURL, a.GetAnthropicProtocolBaseURL())
	for _, capability := range []OpenAIEndpointCapability{OpenAIEndpointCapabilityChatCompletions, OpenAIEndpointCapabilityResponses, OpenAIEndpointCapabilityEmbeddings, OpenAIEndpointCapabilityEmbeddingsMultimodal, OpenAIEndpointCapabilitySeedance} {
		require.True(t, a.SupportsOpenAIEndpointCapability(capability), capability)
	}
	require.True(t, a.SupportsOpenAIImageCapability(OpenAIImagesCapabilityAPIKey))
	a.Type = AccountTypeOAuth
	require.False(t, a.SupportsOpenAIImageCapability(OpenAIImagesCapabilityAPIKey))
	require.False(t, a.SupportsOpenAIEndpointCapability(OpenAIEndpointCapabilitySeedance))
	require.False(t, a.SupportsOpenAIEndpointCapability(OpenAIEndpointCapabilityEmbeddingsMultimodal))
	require.False(t, (&Account{Platform: PlatformOpenAI, Type: AccountTypeAPIKey}).SupportsOpenAIEndpointCapability(OpenAIEndpointCapabilityEmbeddingsMultimodal))
}

func TestArkModelDiscovery(t *testing.T) {
	body := []byte(`{"data":[{"id":"doubao-seed-2-1-pro-260915","domain":"VLM","status":"Published","api_supported":{"chat/completions":true,"responses":true,"messages":true},"modalities":{"input_modalities":["text","image","video"],"output_modalities":["text"]},"features":{"tools":{"function_calling":true},"thinking":{"types":["enabled","disabled","auto"]}},"token_limits":{"max_input_tokens":200000}},{"id":"doubao-seedream-4-0-250828","domain":"ImageGeneration","status":"Retiring","api_supported":{"images/generations":true}},{"id":"kimi-k3-1-pro-261024","api_supported":{"chat/completions":true,"responses":true,"messages":false}}]}`)
	protocols, err := parseModelProtocolCatalog(body)
	require.NoError(t, err)
	require.Equal(t, []string{APIProtocolChatCompletions, APIProtocolResponses, APIProtocolAnthropic}, protocols["doubao-seed-2-1-pro-260915"])
	require.Equal(t, []string{APIProtocolChatCompletions, APIProtocolResponses}, protocols["kimi-k3-1-pro-261024"])
	require.NotContains(t, protocols, "doubao-seedream-4-0-250828")
	ids, metadata, err := extractUpstreamModelCatalog(body, false)
	require.NoError(t, err)
	require.Len(t, ids, 3)
	m := metadata[ids[0]]
	require.Equal(t, []string{"text", "image", "video"}, m.InputModalities)
	require.Equal(t, []string{"text"}, m.OutputModalities)
	require.True(t, *m.FunctionCalling)
	require.True(t, *m.Reasoning)
	require.JSONEq(t, `200000`, string(m.TokenLimits["max_input_tokens"]))
	require.Equal(t, "Retiring", metadata[ids[1]].Status)
	require.Equal(t, "ImageGeneration", metadata[ids[1]].Domain)
}

func TestArkSyncPersistsSparseAndMediaModels(t *testing.T) {
	for _, failSave := range []bool{false, true} {
		t.Run(map[bool]string{false: "complete_catalog", true: "save_failure"}[failSave], func(t *testing.T) {
			upstream := &httpUpstreamRecorder{resp: &http.Response{
				StatusCode: http.StatusOK, Header: http.Header{"Content-Type": []string{"application/json"}},
				Body: io.NopCloser(strings.NewReader(`{"data":[{"id":"doubao-seedream-5-5-261028","domain":"ImageGeneration","status":"Published"},{"id":"hyper3d-gen2-260112"},{"id":"doubao-embedding-vision-251215","domain":"Embedding"},{"id":"doubao-seed-translation-250915","status":"Published"}]}`)),
			}}
			repo := &upstreamModelMetadataRepoStub{}
			if failSave {
				repo.err = errors.New("storage unavailable")
			}
			svc := &AccountTestService{accountRepo: repo, httpUpstream: upstream, cfg: upstreamModelSyncTestConfig()}
			account := arkTestAccount()
			account.SetUpstreamModelMetadataSnapshot(UpstreamModelMetadataSnapshot{Models: map[string]UpstreamModelMetadata{"removed-model": {ID: "removed-model"}}})
			catalog, err := svc.SyncUpstreamModelCatalog(context.Background(), account)
			require.Len(t, upstream.requests, 1, "Ark sync must not depend on the agent metadata registry")
			if failSave {
				require.Error(t, err)
				require.Contains(t, account.GetUpstreamModelMetadataSnapshot().Models, "removed-model")
				return
			}
			require.NoError(t, err)
			require.Empty(t, catalog.Warnings)
			snapshot := account.GetUpstreamModelMetadataSnapshot()
			require.Len(t, snapshot.Models, 4)
			require.NotContains(t, snapshot.Models, "removed-model")
			for _, id := range catalog.Models {
				require.Equal(t, id, snapshot.Models[id].ID)
			}
			require.Equal(t, "ImageGeneration", snapshot.Models["doubao-seedream-5-5-261028"].Domain)
			require.NotNil(t, repo.updates[UpstreamModelMetadataExtraKey])
		})
	}
}

func TestArkNativeTextProtocolRouting(t *testing.T) {
	for _, ingress := range routingMatrixIngresses() {
		t.Run(ingress.name, func(t *testing.T) {
			account := arkTestAccount()
			// Keep discovery offline: use the known bootstrap rules, and capture the
			// actual gateway request rather than only testing URL helpers.
			key := modelProtocolCatalogKey(account, buildOpenAIModelsURL(DefaultArkBaseURL))
			upstreamModelProtocols.store(key, nil, errors.New("offline test"), time.Now())
			upstream := &httpUpstreamRecorder{err: errors.New("stop after capture")}
			svc := &OpenAIGatewayService{cfg: rawChatCompletionsTestConfig(), httpUpstream: upstream}
			body := routingMatrixCase{ingress: ingress, model: "doubao-seed-2-1-pro-260915"}.body()
			_ = ingress.forward(svc, adaptiveProtocolTestContext(ingress.path, body), account, body)
			require.NotNil(t, upstream.lastReq)
			switch ingress.name {
			case "chat":
				require.Equal(t, DefaultArkBaseURL+"/chat/completions", upstream.lastReq.URL.String())
			case "responses", "chat_responses_shape":
				require.Equal(t, DefaultArkBaseURL+"/responses", upstream.lastReq.URL.String())
			case "messages":
				require.Equal(t, DefaultArkAnthropicBaseURL+"/v1/messages", upstream.lastReq.URL.String())
				require.Equal(t, "test-ark-key", upstream.lastReq.Header.Get("x-api-key"))
			default:
				t.Fatalf("unexpected ingress %s", ingress.name)
			}
		})
	}
}

func TestArkTranslationUsesResponsesAndPreservesOptions(t *testing.T) {
	account := arkTestAccount()
	key := modelProtocolCatalogKey(account, buildOpenAIModelsURL(DefaultArkBaseURL))
	upstreamModelProtocols.store(key, nil, errors.New("offline test"), time.Now())
	upstream := &httpUpstreamRecorder{err: errors.New("stop after capture")}
	svc := &OpenAIGatewayService{cfg: rawChatCompletionsTestConfig(), httpUpstream: upstream}
	body := []byte(`{"model":"doubao-seed-translation-250915","input":[{"role":"user","content":[{"type":"input_text","text":"你好","translation_options":{"source_language":"zh","target_language":"en"}}]}]}`)
	for _, ingress := range routingMatrixIngresses() {
		if ingress.name != "responses" {
			continue
		}
		_ = ingress.forward(svc, adaptiveProtocolTestContext(ingress.path, body), account, body)
		require.NotNil(t, upstream.lastReq)
		require.Equal(t, DefaultArkBaseURL+"/responses", upstream.lastReq.URL.String())
		require.Equal(t, gjson.GetBytes(body, "input.0.content.0.translation_options").Raw, gjson.GetBytes(upstream.lastBody, "input.0.content.0.translation_options").Raw)
		require.False(t, gjson.GetBytes(upstream.lastBody, "store").Exists(), "translation must not acquire OpenAI store defaults")
	}
}

func TestArkResponsesPreservesClientStateFields(t *testing.T) {
	body := []byte(`{"model":"doubao-seed-2-1-lite-260915","store":true,"previous_response_id":"resp_previous","input":"Continue"}`)
	require.Equal(t, body, normalizeDeepSeekResponsesRequestBody(arkTestAccount(), body))
}

func TestArkImagesPreserveNativeFieldsAndCountSuccessfulOutputs(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "json", true: "sse"}[stream], func(t *testing.T) {
			body := []byte(`{"model":"art","prompt":"a landscape","image":["https://example.com/ref.png"],"size":"2K","watermark":false,"sequential_image_generation":"auto","sequential_image_generation_options":{"max_images":4},"future_field":{"keep":true},"stream":` + map[bool]string{false: "false", true: "true"}[stream] + `}`)
			response := `{"data":[{"url":"https://example.com/one.png","size":"2048x2048"},{"error":{"code":"failed"}},{"url":"https://example.com/two.png","size":"2048x2048"}],"usage":{"generated_images":2}}`
			contentType := "application/json"
			if stream {
				contentType = "text/event-stream"
				response = "data: {\"type\":\"image_generation.partial_succeeded\",\"image_index\":0,\"url\":\"https://example.com/one.png\",\"size\":\"2048x2048\"}\n\ndata: {\"type\":\"image_generation.partial_failed\",\"image_index\":1,\"error\":{\"code\":\"failed\"}}\n\ndata: {\"type\":\"image_generation.partial_succeeded\",\"image_index\":2,\"url\":\"https://example.com/two.png\",\"size\":\"2048x2048\"}\n\ndata: {\"type\":\"image_generation.completed\",\"usage\":{\"generated_images\":2}}\n\n"
			}
			c, w := grokMediaContentTestContext(http.MethodPost, "/api/v3/images/generations", nil)
			c.Request.Header.Set("Content-Type", "application/json")
			c.Set("api_key", &APIKey{Group: &Group{Platform: PlatformArk}})
			upstream := &httpUpstreamRecorder{resp: &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{contentType}}, Body: io.NopCloser(strings.NewReader(response))}}
			svc := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
			account := arkTestAccount()
			account.Credentials["model_mapping"] = map[string]any{"art": "doubao-seedream-5-5-261028"}
			parsed, err := svc.ParseOpenAIImagesRequest(c, body)
			require.NoError(t, err)
			require.Equal(t, OpenAIImagesCapabilityAPIKey, parsed.RequiredCapability)
			require.Equal(t, []string{"https://example.com/ref.png"}, parsed.InputImageURLs)
			result, err := svc.ForwardImages(context.Background(), c, account, body, parsed, "")
			require.NoError(t, err)
			require.Equal(t, 2, result.ImageCount)
			require.True(t, result.ArkImage)
			require.Equal(t, DefaultArkBaseURL+"/images/generations", upstream.lastReq.URL.String())
			require.Equal(t, "Bearer test-ark-key", upstream.lastReq.Header.Get("Authorization"))
			for _, field := range []string{"image", "size", "watermark", "sequential_image_generation", "sequential_image_generation_options", "future_field", "stream"} {
				require.Equal(t, gjson.GetBytes(body, field).Raw, gjson.GetBytes(upstream.lastBody, field).Raw, field)
			}
			require.Equal(t, "doubao-seedream-5-5-261028", gjson.GetBytes(upstream.lastBody, "model").String())
			require.Contains(t, w.Body.String(), "https://example.com/one.png")
			require.Contains(t, w.Body.String(), "https://example.com/two.png")
		})
	}
}

func TestArkEmbeddingsNativeObjectResponse(t *testing.T) {
	for _, suffix := range []string{"/embeddings", "/embeddings/multimodal"} {
		t.Run(suffix, func(t *testing.T) {
			body := []byte(`{"model":"doubao-embedding-vision-251215","input":[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"https://example.com/ref.png"}}],"dimensions":1024}`)
			response := `{"data":{"embedding":[0.1,0.2]},"usage":{"prompt_tokens":15,"prompt_tokens_details":{"text_tokens":5,"image_tokens":10}}}`
			upstream := &httpUpstreamRecorder{resp: &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: io.NopCloser(strings.NewReader(response))}}
			svc := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
			c, w := grokMediaContentTestContext(http.MethodPost, "/api/v3"+suffix, nil)
			result, err := svc.ForwardEmbeddings(context.Background(), c, arkTestAccount(), body, "")
			require.NoError(t, err)
			require.JSONEq(t, response, w.Body.String())
			require.JSONEq(t, string(body), string(upstream.lastBody))
			require.Equal(t, DefaultArkBaseURL+suffix, upstream.lastReq.URL.String())
			require.Equal(t, 15, result.Usage.InputTokens)
			require.Equal(t, 10, result.Usage.ImageInputTokens)
		})
	}
}

func TestArk3DTasksUseDefaultBaseAndTokenUsage(t *testing.T) {
	for _, model := range []string{"doubao-seed3d-2-0-260328", "hitem3d-2-0-251223", "hyper3d-gen2-260112"} {
		t.Run(model, func(t *testing.T) {
			upstream := &grokMediaContentUpstreamStub{response: grokMediaContentStatusResponse(`{"id":"three-d-task"}`)}
			svc := &OpenAIGatewayService{httpUpstream: upstream}
			c, _ := grokMediaContentTestContext(http.MethodPost, "/api/v3/contents/generations/tasks", nil)
			account := arkTestAccount()
			body := []byte(`{"model":"` + model + `","content":[{"type":"image_url","image_url":{"url":"https://example.com/ref.png"}}]}`)
			created, err := svc.ForwardSeedance(context.Background(), c, account, SeedanceEndpointCreate, "", body)
			require.NoError(t, err)
			require.Equal(t, "seedance:three-d-task", created.ResponseID)
			require.Equal(t, DefaultArkBaseURL+"/contents/generations/tasks", upstream.request.URL.String())
			response := `{"id":"three-d-task","status":"succeeded","model":"` + model + `","content":{"file_url":"https://example.com/model.glb"},"usage":{"completion_tokens":30000}}`
			upstream.response = grokMediaContentStatusResponse(response)
			c, w := grokMediaContentTestContext(http.MethodGet, "/api/v3/contents/generations/tasks/three-d-task", nil)
			result, err := svc.ForwardSeedance(context.Background(), c, account, SeedanceEndpointStatus, created.ResponseID, nil)
			require.NoError(t, err)
			require.Equal(t, 30000, result.Usage.OutputTokens)
			require.Zero(t, result.VideoCount)
			require.JSONEq(t, response, w.Body.String())
		})
	}
	require.Equal(t, 15*24*time.Hour, mediaTaskPendingTTL(nil, "seedance:task"))
	require.Equal(t, 16*24*time.Hour, mediaTaskClaimTTL(nil, "seedance:task"))
	require.Equal(t, 24*time.Hour, mediaTaskPendingTTL(nil, "grok-task"))
	require.Equal(t, 48*time.Hour, mediaTaskClaimTTL(nil, "grok-task"))
}

func TestArkImagePricingRequiresOperatorPrice(t *testing.T) {
	svc := &OpenAIGatewayService{billingService: &BillingService{}}
	key := &APIKey{Group: &Group{Platform: PlatformArk}}
	result := &OpenAIForwardResult{ArkImage: true, ImageEndpoint: true, ImageCount: 2, ImageSize: "2K"}
	calculate := func() (*CostBreakdown, error) {
		return svc.calculateOpenAIRecordUsageCost(context.Background(), result, key, []string{"doubao-seedream-5-5-261028"}, 1, 2, 1, 1, UsageTokens{}, "", nil, false, time.Time{})
	}
	_, err := calculate()
	require.ErrorIs(t, err, ErrModelPricingUnavailable)
	key.Group.ImagePrice2K = f64p(0.2)
	cost, err := calculate()
	require.NoError(t, err)
	require.InDelta(t, 0.4, cost.TotalCost, 0.000001)
	require.InDelta(t, 0.8, cost.ActualCost, 0.000001)
	result.ImageCount = 0
	key.Group.ImagePrice2K = nil
	cost, err = calculate()
	require.NoError(t, err)
	require.Zero(t, cost.TotalCost)
}

func TestArkImageEventsUseIndexAndIgnoreFailures(t *testing.T) {
	counter := newOpenAIImageOutputCounter()
	for _, event := range []string{
		`{"type":"image_generation.partial_succeeded","image_index":0,"url":"https://example.com/same.png","size":"2048x2048"}`,
		`{"type":"image_generation.partial_succeeded","image_index":0,"url":"https://example.com/same.png","size":"2048x2048"}`,
		`{"type":"image_generation.partial_succeeded","image_index":1,"url":"https://example.com/same.png","size":"2048x2048"}`,
		`{"type":"image_generation.partial_failed","image_index":2,"error":{"code":"failed"}}`,
		`{"type":"image_generation.completed","usage":{"generated_images":2}}`,
	} {
		counter.AddSSEData([]byte(event))
	}
	require.Equal(t, 2, counter.Count())
	require.Len(t, counter.Sizes(), 2)
}

func TestArkImagesRejectOpenAIEditsAndAllowEndpointAliases(t *testing.T) {
	svc := &OpenAIGatewayService{}
	body := []byte(`{"model":"art","prompt":"change background","image":"https://example.com/ref.png"}`)
	c, _ := grokMediaContentTestContext(http.MethodPost, "/v1/images/edits", nil)
	c.Request.Header.Set("Content-Type", "application/json")
	c.Set("api_key", &APIKey{Group: &Group{Platform: PlatformArk}})
	_, err := svc.ParseOpenAIImagesRequest(c, body)
	require.ErrorContains(t, err, "JSON /images/generations")
	c, _ = grokMediaContentTestContext(http.MethodPost, "/v1/images/generations", nil)
	c.Request.Header.Set("Content-Type", "application/json")
	c.Set("api_key", &APIKey{Group: &Group{Platform: PlatformArk}})
	parsed, err := svc.ParseOpenAIImagesRequest(c, body)
	require.NoError(t, err)
	a := arkTestAccount()
	a.Credentials["model_mapping"] = map[string]any{"art": "ep-art"}
	a.Credentials["api_protocol"] = APIProtocolAnthropic
	a.Credentials["base_url"] = DefaultArkAnthropicBaseURL
	upstream := &httpUpstreamRecorder{resp: &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: io.NopCloser(strings.NewReader(`{"data":[{"url":"https://example.com/output.png"}]}`))}}
	svc.httpUpstream = upstream
	result, err := svc.ForwardImages(context.Background(), c, a, body, parsed, "")
	require.NoError(t, err)
	require.Equal(t, "ep-art", result.UpstreamModel)
	require.Equal(t, DefaultArkBaseURL+"/images/generations", upstream.lastReq.URL.String())
}

type arkConnectionAccountRepo struct {
	AccountRepository
	account *Account
}

func (r *arkConnectionAccountRepo) GetByID(context.Context, int64) (*Account, error) {
	return r.account, nil
}

func TestArkAdminConnectionTestUsesImagesEndpoint(t *testing.T) {
	a := arkTestAccount()
	a.Credentials["model_mapping"] = map[string]any{"art": "doubao-seedream-5-5-261028"}
	upstream := &httpUpstreamRecorder{resp: &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: io.NopCloser(strings.NewReader(`{"data":[{"url":"https://example.com/generated.png"}]}`))}}
	svc := &AccountTestService{accountRepo: &arkConnectionAccountRepo{account: a}, httpUpstream: upstream, cfg: &config.Config{}}
	c, w := grokMediaContentTestContext(http.MethodPost, "/api/v1/admin/accounts/84/test", nil)
	err := svc.TestAccountConnection(c, a.ID, "art", "a lake", "")
	require.NoError(t, err)
	require.Equal(t, DefaultArkBaseURL+"/images/generations", upstream.lastReq.URL.String())
	require.Equal(t, "Bearer test-ark-key", upstream.lastReq.Header.Get("Authorization"))
	require.Equal(t, "doubao-seedream-5-5-261028", gjson.GetBytes(upstream.lastBody, "model").String())
	require.Equal(t, "2K", gjson.GetBytes(upstream.lastBody, "size").String())
	require.False(t, gjson.GetBytes(upstream.lastBody, "n").Exists())
	require.Contains(t, w.Body.String(), "https://example.com/generated.png")
	require.Contains(t, w.Body.String(), `"success":true`)
}
