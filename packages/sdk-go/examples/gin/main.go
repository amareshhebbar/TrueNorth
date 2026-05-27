// Example: TrueNorth with Gin web framework
package main

import (
	"context"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
	truenorth "github.com/studioilios/truenorth"
)

var tn = truenorth.New(
	os.Getenv("TRUENORTH_API_URL"),
	os.Getenv("TRUENORTH_API_KEY"),
)

func main() {
	r := gin.Default()

	r.POST("/chat/start", func(c *gin.Context) {
		var req struct{ GoalID string `json:"goal_id"` }
		c.ShouldBindJSON(&req)
		session, err := tn.StartSession(context.Background(), req.GoalID, "")
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"session_id": session.SessionID, "message": session.WelcomeMessage})
	})

	r.POST("/chat/:id", func(c *gin.Context) {
		var req struct{ Message string `json:"message"` }
		c.ShouldBindJSON(&req)
		resp, err := tn.SendMessage(context.Background(), c.Param("id"), req.Message)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, resp)
	})

	r.Run(":3000")
}
