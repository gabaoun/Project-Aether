package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/go-redis/redis/v8"
	"github.com/gorilla/websocket"
)

var ctx = context.Background()
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins for the demo
	},
}

// Client represents a connected WebSocket client
type Client struct {
	conn *websocket.Conn
	send chan []byte
}

var (
	clients    = make(map[*Client]bool)
	register   = make(chan *Client)
	unregister = make(chan *Client)
	broadcast  = make(chan []byte)
)

func main() {
	log.Println("Starting Project Aether WebSocket Service (Mensageiro)...")

	// 1. Setup Redis Client to listen for events from the AWS Lambda
	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password set
		DB:       0,  // use default DB
	})

	// 2. Start the internal client hub
	go runHub()

	// 3. Subscribe to Redis Channel
	go subscribeToRedisEvents(rdb)

	// 4. Expose WebSocket Endpoint
	http.HandleFunc("/ws", serveWs)

	log.Println("WebSocket Service listening on :8081")
	if err := http.ListenAndServe(":8081", nil); err != nil {
		log.Fatalf("WebSocket Service Failed: %v", err)
	}
}

func runHub() {
	for {
		select {
		case client := <-register:
			clients[client] = true
			log.Println("New client connected.")
		case client := <-unregister:
			if _, ok := clients[client]; ok {
				delete(clients, client)
				close(client.send)
				log.Println("Client disconnected.")
			}
		case message := <-broadcast:
			for client := range clients {
				select {
				case client.send <- message:
				default:
					close(client.send)
					delete(clients, client)
				}
			}
		}
	}
}

func subscribeToRedisEvents(rdb *redis.Client) {
	pubsub := rdb.Subscribe(ctx, "aether_events")
	defer pubsub.Close()

	ch := pubsub.Channel()

	log.Println("Subscribed to Redis channel 'aether_events'. Waiting for Lambda notifications...")
	for msg := range ch {
		// When Python/Lambda finishes, it sends a message here. We broadcast it to all WS clients.
		log.Printf("Received event from Redis: %s\n", msg.Payload)
		broadcast <- []byte(fmt.Sprintf(`{"type": "system_notification", "message": "%s"}`, msg.Payload))
	}
}

func serveWs(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println(err)
		return
	}

	client := &Client{conn: conn, send: make(chan []byte, 256)}
	register <- client

	// Start a goroutine to write messages to the WebSocket connection
	go writePump(client)
}

func writePump(c *Client) {
	defer func() {
		c.conn.Close()
	}()
	for {
		message, ok := <-c.send
		if !ok {
			c.conn.WriteMessage(websocket.CloseMessage, []byte{})
			return
		}
		c.conn.WriteMessage(websocket.TextMessage, message)
	}
}
