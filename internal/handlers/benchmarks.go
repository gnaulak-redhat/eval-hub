package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/eval-hub/eval-hub/internal/executioncontext"
	"github.com/eval-hub/eval-hub/pkg/api"
)

// HandleListBenchmarks handles GET /api/v1/evaluations/benchmarks
func (h *Handlers) HandleListBenchmarks(ctx *executioncontext.ExecutionContext, w http.ResponseWriter) {
	if !h.checkMethod(ctx, http.MethodGet, w) {
		return
	}

	benchmarks := []api.BenchmarkResource{}
	for _, provider := range ctx.ProviderConfigs {
		for _, benchmark := range provider.Benchmarks {
			benchmark.ProviderId = &provider.ProviderID
			benchmarks = append(benchmarks, benchmark)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(api.BenchmarkResourceList{
		TotalCount: len(benchmarks),
		Items:      benchmarks,
	})
}
