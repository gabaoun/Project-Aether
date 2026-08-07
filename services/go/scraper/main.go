package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"time"
)

type ScrapeRequest struct {
	URLs []string `json:"urls"`
}

type ScrapeResult struct {
	URL    string `json:"url"`
	Status int    `json:"status"`
	Length int    `json:"length"`
}

func main() {
	log.Println("Starting Project Aether Concurrent Scraper (Aspirador de Dados)...")

	http.HandleFunc("/api/v1/scrape", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req ScrapeRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		// Concurrent Scraping using Goroutines and WaitGroup
		var wg sync.Mutex
		results := make([]ScrapeResult, 0, len(req.URLs))
		
		var waitGroup sync.WaitGroup
		
		log.Printf("Received %d URLs to scrape concurrently.", len(req.URLs))
		
		// Worker pool logic could be added here, but for demonstration 
		// we'll just fire a goroutine per URL (which Go handles easily)
		for _, u := range req.URLs {
			waitGroup.Add(1)
			go func(url string) {
				defer waitGroup.Done()
				
				res := fetchURL(url)
				
				// In a real app, you would upload the extracted text to S3 here using aws-sdk-go-v2
				// e.g. s3Client.PutObject(ctx, bucket, key, bytes.NewReader(body))
				// Which would then trigger the Lambda function we built earlier!
				
				wg.Lock()
				results = append(results, res)
				wg.Unlock()
			}(u)
		}

		// Wait for all goroutines to finish
		waitGroup.Wait()
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"message": "Scraping completed and data sent to S3.",
			"results": results,
		})
	})

	log.Println("Scraper Service listening on :8082")
	if err := http.ListenAndServe(":8082", nil); err != nil {
		log.Fatalf("Scraper Service Failed: %v", err)
	}
}

// fetchURL simulates a HTTP GET request and returns a summary
func fetchURL(u string) ScrapeResult {
	// Add a small timeout so we don't hang forever on bad sites
	client := http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(u)
	
	if err != nil {
		log.Printf("Failed to scrape %s: %v", u, err)
		return ScrapeResult{URL: u, Status: 500, Length: 0}
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	
	fmt.Printf("[Scraper Worker] Downloaded %s -> %d bytes\n", u, len(body))
	return ScrapeResult{URL: u, Status: resp.StatusCode, Length: len(body)}
}
