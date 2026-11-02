package openai

import (
	"strings"
	"testing"
)

func TestParseChat(t *testing.T) {
	raw := `{
		"model": "qwen",
		"stream": true,
		"stream_options": {"include_usage": true},
		"max_tokens": 32,
		"messages": [
			{"role": "system", "content": "You are terse."},
			{"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "image_url", "image_url": {"url": "x"}}]}
		]
	}`
	req, err := Parse("/v1/chat/completions", []byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	if req.Model != "qwen" || !req.Stream || !req.IncludeUsage || req.MaxTokens != 32 {
		t.Errorf("unexpected fields: %+v", req)
	}
	want := "system\x00You are terse.\x00user\x00Hi\x00"
	if req.Prompt != want {
		t.Errorf("Prompt = %q, want %q", req.Prompt, want)
	}
}

func TestParseChatPromptsSharePrefix(t *testing.T) {
	a, _ := Parse("/v1/chat/completions", []byte(`{"messages":[{"role":"system","content":"doc"},{"role":"user","content":"q1"}]}`))
	b, _ := Parse("/v1/chat/completions", []byte(`{"messages":[{"role":"system","content":"doc"},{"role":"user","content":"q2"}]}`))
	if !strings.HasPrefix(b.Prompt, a.Prompt[:len(a.Prompt)-3]) {
		t.Errorf("prompts with the same system message do not share a prefix: %q, %q", a.Prompt, b.Prompt)
	}
}

func TestParseCompletion(t *testing.T) {
	req, err := Parse("/v1/completions", []byte(`{"prompt": "Once upon", "max_completion_tokens": 8}`))
	if err != nil {
		t.Fatal(err)
	}
	if req.Prompt != "Once upon" || req.MaxTokens != 8 {
		t.Errorf("unexpected request: %+v", req)
	}
}

func TestParseErrors(t *testing.T) {
	cases := map[string]struct{ path, body string }{
		"bad json":     {"/v1/chat/completions", `{`},
		"no messages":  {"/v1/chat/completions", `{"messages": []}`},
		"bad content":  {"/v1/chat/completions", `{"messages": [{"role": "user", "content": 5}]}`},
		"unknown path": {"/v1/embeddings", `{}`},
		"bad prompt":   {"/v1/completions", `{"prompt": {"a": 1}}`},
	}
	for name, c := range cases {
		if _, err := Parse(c.path, []byte(c.body)); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}
