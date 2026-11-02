// Package openai parses the parts of OpenAI-compatible completion requests
// that routing needs, and writes errors in the OpenAI format.
package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// Request is what the router reads from a completion request. The original
// body is forwarded unchanged.
type Request struct {
	Model        string
	Stream       bool
	IncludeUsage bool
	MaxTokens    int
	// Prompt is the text a backend runs prefill over, flattened in message
	// order. Two requests that share a prompt prefix share a Prompt prefix.
	Prompt string
}

type message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type contentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type body struct {
	Model               string          `json:"model"`
	Stream              bool            `json:"stream"`
	MaxTokens           int             `json:"max_tokens"`
	MaxCompletionTokens int             `json:"max_completion_tokens"`
	Messages            []message       `json:"messages"`
	Prompt              json.RawMessage `json:"prompt"`
	StreamOptions       struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
}

// Parse reads a request to /v1/chat/completions or /v1/completions.
func Parse(path string, raw []byte) (Request, error) {
	var b body
	if err := json.Unmarshal(raw, &b); err != nil {
		return Request{}, fmt.Errorf("invalid JSON body: %w", err)
	}
	req := Request{
		Model:        b.Model,
		Stream:       b.Stream,
		IncludeUsage: b.StreamOptions.IncludeUsage,
		MaxTokens:    b.MaxTokens,
	}
	if b.MaxCompletionTokens > 0 {
		req.MaxTokens = b.MaxCompletionTokens
	}

	var sb strings.Builder
	switch {
	case strings.HasSuffix(path, "/chat/completions"):
		if len(b.Messages) == 0 {
			return Request{}, errors.New("messages must not be empty")
		}
		for _, m := range b.Messages {
			text, err := contentText(m.Content)
			if err != nil {
				return Request{}, err
			}
			// A separator keeps "ab"+"c" and "a"+"bc" apart.
			sb.WriteString(m.Role)
			sb.WriteByte(0)
			sb.WriteString(text)
			sb.WriteByte(0)
		}
	case strings.HasSuffix(path, "/completions"):
		text, err := contentText(b.Prompt)
		if err != nil {
			return Request{}, err
		}
		sb.WriteString(text)
	default:
		return Request{}, fmt.Errorf("unsupported path %q", path)
	}
	req.Prompt = sb.String()
	return req, nil
}

// contentText accepts a string or an array of content parts and returns the
// text parts joined. Images and other parts are skipped.
func contentText(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, nil
	}
	var parts []contentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", errors.New("content must be a string or an array of parts")
	}
	var sb strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			sb.WriteString(p.Text)
		}
	}
	return sb.String(), nil
}

// Usage is the token accounting a backend returns.
type Usage struct {
	PromptTokens        int                  `json:"prompt_tokens"`
	CompletionTokens    int                  `json:"completion_tokens"`
	TotalTokens         int                  `json:"total_tokens"`
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`
}

// PromptTokensDetails says how much of the prompt the backend took from its
// KV cache.
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// WriteError writes an error body in the shape OpenAI clients expect.
func WriteError(w http.ResponseWriter, status int, typ, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": msg, "type": typ},
	})
}
