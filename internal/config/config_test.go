package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func write(t *testing.T, text string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "router.yaml")
	if err := os.WriteFile(path, []byte(text), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadKeepsDefaults(t *testing.T) {
	cfg, err := Load(write(t, `
policy: least_loaded
backends:
  - {name: a, url: "http://127.0.0.1:8081", max_inflight: 4}
health:
  interval: 500ms
`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Policy != "least_loaded" || cfg.Backends[0].MaxInflight != 4 {
		t.Errorf("file values not applied: %+v", cfg)
	}
	if cfg.Health.Interval != 500*time.Millisecond {
		t.Errorf("interval = %v, want 500ms", cfg.Health.Interval)
	}
	if cfg.Health.Path != "/health" || cfg.Prefix.ChunkSize != 256 || cfg.Listen != ":8080" {
		t.Errorf("defaults lost: %+v", cfg)
	}
}

func TestExampleConfigIsValid(t *testing.T) {
	if _, err := Load("../../config.example.yaml"); err != nil {
		t.Fatal(err)
	}
}

func TestValidate(t *testing.T) {
	cases := map[string]string{
		"no backends":    `backends: []`,
		"duplicate name": `backends: [{name: a, url: "http://x:1"}, {name: a, url: "http://y:1"}]`,
		"bad url":        `backends: [{name: a, url: "localhost:8080"}]`,
		"bad policy":     "policy: random\nbackends: [{name: a, url: \"http://x:1\"}]",
		"bad ratio":      "prefix: {min_match_ratio: 2}\nbackends: [{name: a, url: \"http://x:1\"}]",
	}
	for name, text := range cases {
		_, err := Load(write(t, text))
		if err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	if _, err := Load(write(t, "listen: [")); err == nil || !strings.Contains(err.Error(), "parse") {
		t.Errorf("broken YAML: err = %v", err)
	}
}
