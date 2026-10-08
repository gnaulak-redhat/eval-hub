package api

// OCISigningConfig selects the signing method and optionally references a key Secret.
// It retains configuration only; it does not resolve keys or schedule signing.
type OCISigningConfig struct {
	Type      string `json:"type" validate:"required,oneof=key-pair"`
	SecretRef string `json:"secret_ref,omitempty" validate:"omitempty,notblank"`
}

// OCIArtifactReference identifies an OCI artifact by digest.
type OCIArtifactReference struct {
	OCIDigest    string `json:"oci_digest" validate:"required,notblank"`
	OCIReference string `json:"oci_reference" validate:"required,notblank"`
}

// OCISigningState is independent of the evaluation job state.
type OCISigningState string

const (
	OCISigningPending OCISigningState = "pending"
	OCISigningSigned  OCISigningState = "signed"
	OCISigningFailed  OCISigningState = "failed"
)

type OCISigningStatus struct {
	State   OCISigningState `json:"state" validate:"required,oneof=pending signed failed"`
	Message *MessageInfo    `json:"message" validate:"required"`
}

// OCISigningResult reports an actual signing outcome; it has no default state.
type OCISigningResult struct {
	Status *OCISigningStatus `json:"status" validate:"required"`
	Type   string            `json:"type" validate:"required,oneof=key-pair"`
}

// EvaluationOCIResults remains unset until artifact and signing workflows supply values.
type EvaluationOCIResults struct {
	EvaluationCard   *OCIArtifactReference `json:"evaluation_card,omitempty"`
	EvaluationBundle *OCIArtifactReference `json:"evaluation_bundle,omitempty"`
	Signing          *OCISigningResult     `json:"signing,omitempty"`
}
