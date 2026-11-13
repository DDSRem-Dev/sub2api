package service

import "strings"

const (
	DefaultArkBaseURL                                                     = "https://ark.cn-beijing.volces.com/api/v3"
	DefaultArkAnthropicBaseURL                                            = "https://ark.cn-beijing.volces.com/api/compatible"
	OpenAIEndpointCapabilityEmbeddingsMultimodal OpenAIEndpointCapability = "embeddings_multimodal"
)

func (a *Account) IsArk() bool { return a != nil && a.Platform == PlatformArk }

// Ark is an aggregator: model family names must not route its versioned IDs to
// the original vendor. This bootstrap catalog mirrors the 2026-11-13 catalog;
// authenticated /models discovery remains authoritative for availability.
var arkModelIDs = strings.Fields(`
kimi-k3-1-pro-261024
doubao-seedream-5-5-261028
deepseek-v4-1-flash-economy-261017
deepseek-v4-1-flash-261017
doubao-seedream-5-0-flash-260915
doubao-seed-2-1-pro-260915
deepseek-v4-1-flash-260910
glm-5-3-flash-260828
doubao-seed-2-1-lite-260915
deepseek-v4-pro-ga-260813
deepseek-v4-flash-ga-260731
doubao-seedance-2-5-260628
doubao-seedream-5-0-pro-260628
glm-5-2-260617
doubao-seed-evolving
doubao-seed-character-260628
doubao-seed-2-1-turbo-260628
doubao-seed-2-1-pro-260628
doubao-seedance-2-0-mini-260615
deepseek-v4-pro-260425
doubao-seed-2-0-lite-260428
doubao-seed-2-0-mini-260428
doubao-seedream-4-0-20260415
doubao-seed3d-2-0-260328
hitem3d-2-0-251223
hyper3d-gen2-260112
doubao-seed-2-0-code-preview-260215
doubao-seedance-2-0-fast-260128
doubao-seed-2-0-pro-260215
doubao-seedream-5-0-260128
doubao-seedance-2-0-260128
doubao-seed-character-251128
doubao-embedding-vision-251215
doubao-seedream-4-5-251128
qwen3-14b-20250429
qwen3-0-6b-20250429
doubao-seedance-1-0-pro-fast-251015
doubao-smart-router-250928
doubao-seed-translation-250915
doubao-seedream-4-0-250828
qwen2-5-72b-20240919
qwen3-32b-20250429
qwen3-8b-20250429
glm-4-5-air-20250728
doubao-embedding-vision-250615
doubao-seedance-1-0-pro-250528
doubao-embedding-large-text-250515
doubao-seedance-1-0-lite-t2v-250428
doubao-seedance-1-0-lite-i2v-250428
doubao-embedding-vision-250328
doubao-embedding-vision-241215
doubao-embedding-large-text-240915
doubao-embedding-text-240715
doubao-embedding-text-240515
`)

func DefaultArkModelIDs() []string { return append([]string(nil), arkModelIDs...) }

func IsArkModel(model string) bool {
	model = strings.ToLower(strings.TrimSpace(model))
	for _, id := range arkModelIDs {
		if model == id {
			return true
		}
	}
	return strings.HasPrefix(model, "doubao-") || strings.HasPrefix(model, "hyper3d-") || strings.HasPrefix(model, "hitem3d-")
}

func isArkImageModel(model string) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(model)), "doubao-seedream-")
}

// Only known text models receive native-protocol defaults. New models safely
// fall back to Chat until their /models API capabilities have been discovered.
func DefaultArkProtocolRules() []ProtocolRule {
	var rules []ProtocolRule
	for _, id := range arkModelIDs {
		if id == "doubao-seed-translation-250915" {
			rules = append(rules, ProtocolRule{Pattern: id, Protocol: APIProtocolResponses, Protocols: []string{APIProtocolResponses}})
			continue
		}
		if strings.HasPrefix(id, "deepseek-") || strings.HasPrefix(id, "glm-5-") ||
			(strings.HasPrefix(id, "doubao-seed-") && !strings.Contains(id, "translation")) || strings.HasPrefix(id, "kimi-") {
			protocols := []string{APIProtocolChatCompletions, APIProtocolResponses}
			if !strings.HasPrefix(id, "kimi-") && id != "doubao-seed-character-260628" {
				protocols = append(protocols, APIProtocolAnthropic)
			}
			rules = append(rules, ProtocolRule{Pattern: id, Protocol: APIProtocolChatCompletions, Protocols: protocols})
		}
	}
	return rules
}
