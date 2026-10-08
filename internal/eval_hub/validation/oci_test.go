package validation

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/eval-hub/eval-hub/pkg/api"
)

func TestOCISigningRequestContract(t *testing.T) {
	validate := newTestValidator(t)
	for _, tt := range []struct {
		name, signing string
		valid         bool
	}{
		{"omitted", "", true},
		{"configured", `,"signing":{"type":"key-pair","secret_ref":"my-cosign-secret"}`, true},
		{"null", `,"signing":null`, true},
		{"no secret", `,"signing":{"type":"key-pair"}`, true},
		{"missing method", `,"signing":{"secret_ref":"key"}`, false},
		{"null method", `,"signing":{"type":null}`, false},
		{"keyless deferred", `,"signing":{"type":"keyless"}`, false},
		{"empty", `,"signing":{}`, false},
		{"null secret", `,"signing":{"type":"key-pair","secret_ref":null}`, true},
		{"empty secret", `,"signing":{"type":"key-pair","secret_ref":""}`, true},
		{"blank secret", `,"signing":{"type":"key-pair","secret_ref":" \t"}`, false},
		{"wrong type", `,"signing":{"type":"key-pair","secret_ref":42}`, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			raw := `{"oci":{"coordinates":{"oci_host":"quay.io","oci_repository":"org/results"}` + tt.signing + `}}`
			var exports api.EvaluationExports
			err := json.Unmarshal([]byte(raw), &exports)
			if err == nil {
				err = validate.Struct(exports)
			}
			if (err == nil) != tt.valid {
				t.Fatalf("valid=%v, error=%v", tt.valid, err)
			}
			if !tt.valid {
				return
			}
			job := api.EvaluationJobResource{EvaluationJobConfig: api.EvaluationJobConfig{Exports: &exports}}
			encoded, err := json.Marshal(job)
			if err != nil {
				t.Fatal(err)
			}
			var restored api.EvaluationJobResource
			if err := json.Unmarshal(encoded, &restored); err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(encoded), `"results"`) {
				t.Fatalf("configuration produced results: %s", encoded)
			}
			if exports.OCI.Signing == nil {
				if strings.Contains(string(encoded), `"signing"`) {
					t.Fatal("omitted signing was emitted")
				}
			} else {
				if *restored.Exports.OCI.Signing != *exports.OCI.Signing {
					t.Fatal("signing configuration lost")
				}
				if exports.OCI.Signing.SecretRef == "" && strings.Contains(string(encoded), `"secret_ref"`) {
					t.Fatal("absent secret reference emitted")
				}
			}
		})
	}
}

func TestOCIResultsContract(t *testing.T) {
	validate := newTestValidator(t)
	artifact := `{"oci_digest":"sha256:abc","oci_reference":"quay.io/org/results@sha256:abc"}`
	for _, tt := range []struct {
		name, raw string
		valid     bool
	}{
		{"omitted", `{}`, true},
		{"unavailable", `{"oci":{}}`, true},
		{"card", `{"oci":{"evaluation_card":` + artifact + `}}`, true},
		{"bundle", `{"oci":{"evaluation_bundle":` + artifact + `}}`, true},
		{"null oci", `{"oci":null}`, true},
		{"null card", `{"oci":{"evaluation_card":null}}`, true},
		{"null bundle", `{"oci":{"evaluation_bundle":null}}`, true},
		{"null signing", `{"oci":{"signing":null}}`, true},
		{"empty card", `{"oci":{"evaluation_card":{}}}`, false},
		{"incomplete bundle", `{"oci":{"evaluation_bundle":{"oci_digest":"sha256:abc"}}}`, false},
		{"null digest", `{"oci":{"evaluation_card":{"oci_digest":null,"oci_reference":"ref"}}}`, false},
		{"blank reference", `{"oci":{"evaluation_card":{"oci_digest":"digest","oci_reference":" \t"}}}`, false},
		{"empty signing", `{"oci":{"signing":{}}}`, false},
		{"null status", `{"oci":{"signing":{"status":null,"type":"key-pair"}}}`, false},
		{"empty status", `{"oci":{"signing":{"status":{},"type":"key-pair"}}}`, false},
		{"null message", `{"oci":{"signing":{"status":{"state":"signed","message":null},"type":"key-pair"}}}`, false},
		{"empty message", `{"oci":{"signing":{"status":{"state":"signed","message":{}},"type":"key-pair"}}}`, false},
		{"missing message code", `{"oci":{"signing":{"status":{"state":"signed","message":{"message":"Signed"}},"type":"key-pair"}}}`, false},
		{"missing message text", `{"oci":{"signing":{"status":{"state":"signed","message":{"message_code":"signed"}},"type":"key-pair"}}}`, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var results api.EvaluationJobResults
			err := json.Unmarshal([]byte(tt.raw), &results)
			if err == nil {
				err = validate.Struct(results)
			}
			if (err == nil) != tt.valid {
				t.Fatalf("valid=%v, error=%v", tt.valid, err)
			}
			if tt.valid && results.OCI != nil {
				encoded, err := json.Marshal(results.OCI)
				if err != nil {
					t.Fatal(err)
				}
				if strings.Contains(string(encoded), "null") {
					t.Fatalf("null optional object emitted: %s", encoded)
				}
			}
			if tt.name == "omitted" || tt.name == "null oci" {
				data, _ := json.Marshal(results)
				if string(data) != `{}` {
					t.Fatalf("unset results emitted: %s", data)
				}
			}
		})
	}
	for _, state := range []string{"pending", "signed", "failed", "unsigned", ""} {
		for _, method := range []string{"key-pair", "keyless", ""} {
			t.Run(state+"/"+method, func(t *testing.T) {
				result := api.EvaluationOCIResults{Signing: &api.OCISigningResult{
					Type: method, Status: &api.OCISigningStatus{State: api.OCISigningState(state), Message: &api.MessageInfo{Message: "Signing status", MessageCode: "signing_status"}},
				}}
				valid := method == "key-pair" && (state == "pending" || state == "signed" || state == "failed")
				if err := validate.Struct(result); (err == nil) != valid {
					t.Fatalf("valid=%v, error=%v", valid, err)
				}
			})
		}
	}
}

func TestOCISigningStillRequiresCoordinates(t *testing.T) {
	validate := newTestValidator(t)
	for _, raw := range []string{
		`{"signing":{"type":"key-pair","secret_ref":"key"}}`,
		`{"coordinates":{},"signing":{"type":"key-pair","secret_ref":"key"}}`,
	} {
		var exports api.EvaluationExportsOCI
		if err := json.Unmarshal([]byte(raw), &exports); err != nil {
			t.Fatal(err)
		}
		if err := validate.Struct(exports); err == nil {
			t.Fatal("signing configuration bypassed required OCI coordinates")
		}
	}
}
