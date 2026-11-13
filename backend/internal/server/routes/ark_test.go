package routes

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/stretchr/testify/require"
)

func TestArkGatewayAliasesAndPlatformGate(t *testing.T) {
	router := newGatewayRoutesTestRouter(service.PlatformArk)
	registered := map[string]bool{}
	for _, route := range router.Routes() {
		registered[route.Method+" "+route.Path] = true
	}
	for _, prefix := range []string{"/v1", "/v3", "/api/v3"} {
		for _, endpoint := range []string{"/chat/completions", "/responses", "/images/generations", "/embeddings", "/embeddings/multimodal", "/contents/generations/tasks"} {
			require.True(t, registered["POST "+prefix+endpoint], prefix+endpoint)
		}
		require.True(t, registered["GET "+prefix+"/models"])
		require.True(t, registered["GET "+prefix+"/contents/generations/tasks/:task_id"])
		require.True(t, registered["DELETE "+prefix+"/contents/generations/tasks/:task_id"])
	}
	for _, platform := range []string{service.PlatformAnthropic, service.PlatformOpenAI, service.PlatformGrok} {
		t.Run(platform, func(t *testing.T) {
			r := newGatewayRoutesTestRouter(platform)
			for _, path := range []string{"/v1/embeddings/multimodal", "/api/v3/embeddings/multimodal", "/embeddings/multimodal"} {
				w := httptest.NewRecorder()
				req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"model":"doubao-embedding-vision-251215","input":[]}`))
				req.Header.Set("Content-Type", "application/json")
				r.ServeHTTP(w, req)
				require.Equal(t, http.StatusNotFound, w.Code, w.Body.String())
				require.Contains(t, w.Body.String(), "Embeddings API is not supported")
			}
		})
	}
}
