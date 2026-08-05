package main

import (
	"context"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"

	truenorth "github.com/amareshhebbar/truenorth"
)

var tn = truenorth.NewClient(
	os.Getenv("TRUENORTH_API_KEY"),
	os.Getenv("TRUENORTH_API_URL"),
)

func main() {
	r := gin.Default()

	r.POST("/chat/start", func(c *gin.Context) {
		var req struct {
			GoalID string `json:"goal_id"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		session, err := tn.Sessions.Create(context.Background(), req.GoalID, nil)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"session_id": session.ID,
			"message":    session.AgentMessage,
		})
	})

	r.POST("/chat/:id", func(c *gin.Context) {
		var req struct {
			Message string `json:"message"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		resp, err := tn.Sessions.Message(context.Background(), c.Param("id"), req.Message)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, resp)
	})

	r.Run(":3000")
}
