package main

import (
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sync"
	"time"
)

// In-memory rate limiter
type RateLimiter struct {
	mu       sync.Mutex
	visitors map[string]*Visitor
}

type Visitor struct {
	tokens    int
	lastSeen  time.Time
}

func NewRateLimiter() *RateLimiter {
	return &RateLimiter{
		visitors: make(map[string]*Visitor),
	}
}

// Allow limits to 5 requests per second per IP
func (rl *RateLimiter) Allow(ip string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	v, exists := rl.visitors[ip]
	if !exists {
		rl.visitors[ip] = &Visitor{tokens: 5, lastSeen: time.Now()}
		return true
	}

	// Refill 1 token per second
	elapsed := time.Since(v.lastSeen).Seconds()
	if elapsed >= 1 {
		v.tokens += int(elapsed)
		if v.tokens > 5 {
			v.tokens = 5
		}
		v.lastSeen = time.Now()
	}

	if v.tokens > 0 {
		v.tokens--
		return true
	}
	return false
}

func main() {
	log.Println("Starting Project Aether Gateway (Rate Limiter + Reverse Proxy)...")
	
	// Fast API backend URL
	targetURL, _ := url.Parse("http://localhost:8000")
	proxy := httputil.NewSingleHostReverseProxy(targetURL)
	limiter := NewRateLimiter()

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		ip := r.RemoteAddr // In production, use X-Forwarded-For

		if !limiter.Allow(ip) {
			http.Error(w, "Rate Limit Exceeded - Too Many Requests", http.StatusTooManyRequests)
			log.Printf("Blocked %s: Rate Limit Exceeded\n", ip)
			return
		}

		// Proxy valid requests to Python FastAPI
		proxy.ServeHTTP(w, r)
	})

	log.Println("Gateway listening on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("Gateway Failed: %v", err)
	}
}
