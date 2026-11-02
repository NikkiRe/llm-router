// Package config loads the router's YAML configuration.
package config

import (
	"errors"
	"fmt"
	"net/url"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

// Config is the whole router configuration. See config.example.yaml.
type Config struct {
	Listen       string        `yaml:"listen"`
	AdminListen  string        `yaml:"admin_listen"`
	Policy       string        `yaml:"policy"`
	Backends     []Backend     `yaml:"backends"`
	Health       Health        `yaml:"health"`
	Prefix       Prefix        `yaml:"prefix"`
	Limits       Limits        `yaml:"limits"`
	QueueTimeout time.Duration `yaml:"queue_timeout"`
	Retries      int           `yaml:"retries"`
	MaxBodyBytes int64         `yaml:"max_body_bytes"`
	// ResponseHeaderTimeout bounds the wait for a backend's response headers.
	// llama.cpp sends them only when a non-streaming answer is complete, so
	// this must cover a whole generation.
	ResponseHeaderTimeout time.Duration `yaml:"response_header_timeout"`
}

// Backend is one inference server.
type Backend struct {
	Name        string `yaml:"name"`
	URL         string `yaml:"url"`
	APIKey      string `yaml:"api_key"`
	MaxInflight int    `yaml:"max_inflight"`
}

// Health configures active health checks.
type Health struct {
	Path     string        `yaml:"path"`
	Interval time.Duration `yaml:"interval"`
	Timeout  time.Duration `yaml:"timeout"`
	Fall     int           `yaml:"fall"`
	Rise     int           `yaml:"rise"`
}

// Prefix configures cache-aware routing.
type Prefix struct {
	ChunkSize     int     `yaml:"chunk_size"`
	CacheChunks   int     `yaml:"cache_chunks"`
	MinMatchRatio float64 `yaml:"min_match_ratio"`
	LoadSlack     int     `yaml:"load_slack"`
}

// Limits are per API key.
type Limits struct {
	RequestsPerSecond float64  `yaml:"requests_per_second"`
	Burst             int      `yaml:"burst"`
	MaxConcurrent     int      `yaml:"max_concurrent"`
	APIKeys           []string `yaml:"api_keys"`
}

// Default returns the configuration used for every field the file leaves out.
func Default() Config {
	return Config{
		Listen:      ":8080",
		AdminListen: ":9090",
		Policy:      "prefix",
		Health: Health{
			Path:     "/health",
			Interval: 2 * time.Second,
			Timeout:  time.Second,
			Fall:     3,
			Rise:     2,
		},
		Prefix: Prefix{
			ChunkSize:     256,
			CacheChunks:   8192,
			MinMatchRatio: 0.5,
			LoadSlack:     2,
		},
		QueueTimeout:          30 * time.Second,
		Retries:               1,
		MaxBodyBytes:          8 << 20,
		ResponseHeaderTimeout: 5 * time.Minute,
	}
}

// Load reads and validates a YAML file on top of the defaults.
func Load(path string) (Config, error) {
	cfg := Default()
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, err
	}
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("parse %s: %w", path, err)
	}
	return cfg, cfg.Validate()
}

// Validate reports the first problem it finds.
func (c Config) Validate() error {
	if len(c.Backends) == 0 {
		return errors.New("at least one backend is required")
	}
	seen := map[string]bool{}
	for i, b := range c.Backends {
		if b.Name == "" {
			return fmt.Errorf("backend %d: name is required", i)
		}
		if seen[b.Name] {
			return fmt.Errorf("backend name %q is used twice", b.Name)
		}
		seen[b.Name] = true
		u, err := url.Parse(b.URL)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			return fmt.Errorf("backend %q: url must be http(s)://host[:port], got %q", b.Name, b.URL)
		}
		if b.MaxInflight < 0 {
			return fmt.Errorf("backend %q: max_inflight must not be negative", b.Name)
		}
	}
	switch c.Policy {
	case "round_robin", "least_loaded", "prefix":
	default:
		return fmt.Errorf("policy must be round_robin, least_loaded or prefix, got %q", c.Policy)
	}
	if c.Prefix.ChunkSize <= 0 || c.Prefix.CacheChunks <= 0 {
		return errors.New("prefix.chunk_size and prefix.cache_chunks must be positive")
	}
	if c.Prefix.MinMatchRatio < 0 || c.Prefix.MinMatchRatio > 1 {
		return errors.New("prefix.min_match_ratio must be between 0 and 1")
	}
	if c.Health.Interval <= 0 || c.Health.Timeout <= 0 {
		return errors.New("health.interval and health.timeout must be positive")
	}
	if c.QueueTimeout <= 0 {
		return errors.New("queue_timeout must be positive")
	}
	if c.Retries < 0 {
		return errors.New("retries must not be negative")
	}
	return nil
}
