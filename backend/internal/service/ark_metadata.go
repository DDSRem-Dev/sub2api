package service

import (
	"context"
	"encoding/json"
	"time"
)

// Ark's catalog includes images, embeddings and asynchronous generators. Their
// entries must survive sync even without the context/reasoning fields required
// by the Codex agent capability catalog.
func (s *AccountTestService) persistArkModelCatalog(ctx context.Context, account *Account, catalog *UpstreamModelCatalog) (*UpstreamModelCatalog, error) {
	for _, id := range catalog.Models {
		entry := catalog.Metadata[id]
		entry.ID = id
		catalog.Metadata[id] = entry
	}
	catalog.DiscoverySource = "upstream"
	if account.ID <= 0 || s.accountRepo == nil {
		return catalog, nil
	}
	snapshot := UpstreamModelMetadataSnapshot{
		Source: "upstream", SyncedAt: time.Now().UTC().Format(time.RFC3339), Models: catalog.Metadata,
	}
	if err := s.accountRepo.UpdateExtra(ctx, account.ID, map[string]any{UpstreamModelMetadataExtraKey: snapshot}); err != nil {
		return nil, newUpstreamModelSyncInternalError("Failed to save upstream model metadata", err)
	}
	account.SetUpstreamModelMetadataSnapshot(snapshot)
	return catalog, nil
}

// Keep Ark's published capabilities, including non-text modalities and retiring
// status. Discovery is descriptive: it must not silently remove usable models.
func applyArkModelMetadata(raw []byte, entry *UpstreamModelMetadata) {
	var model struct {
		Domain     string          `json:"domain"`
		Status     string          `json:"status"`
		APIs       map[string]bool `json:"api_supported"`
		Modalities struct {
			Input  []string `json:"input_modalities"`
			Output []string `json:"output_modalities"`
		} `json:"modalities"`
		Features struct {
			Tools struct {
				FunctionCalling *bool `json:"function_calling"`
			} `json:"tools"`
			Thinking struct {
				Types []string `json:"types"`
			} `json:"thinking"`
		} `json:"features"`
		TokenLimits map[string]json.RawMessage `json:"token_limits"`
	}
	if entry == nil || json.Unmarshal(raw, &model) != nil {
		return
	}
	entry.Domain = model.Domain
	entry.Status = model.Status
	entry.SupportedAPIs = model.APIs
	entry.OutputModalities = model.Modalities.Output
	entry.FunctionCalling = model.Features.Tools.FunctionCalling
	entry.TokenLimits = model.TokenLimits
	if len(model.Modalities.Input) > 0 {
		entry.InputModalities = model.Modalities.Input
	}
	if len(model.Features.Thinking.Types) > 0 {
		reasoning := false
		for _, kind := range model.Features.Thinking.Types {
			if kind == "enabled" || kind == "auto" {
				reasoning = true
			}
		}
		entry.Reasoning = &reasoning
	}
}
