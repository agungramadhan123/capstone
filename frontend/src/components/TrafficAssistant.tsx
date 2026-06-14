import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { ScrollArea } from './ui/scroll-area'
import { MessageSquare, Send, Bot, User } from 'lucide-react'

export default function TrafficAssistant() {
  const [messages, setMessages] = useState([
    { id: 1, role: 'bot', text: 'Halo! Saya asisten AI Smart Traffic Bandung. Ada yang bisa saya bantu terkait analitik lalu lintas hari ini?' }
  ])
  const [input, setInput] = useState('')

  const handleSend = () => {
    if (!input.trim()) return
    
    // Add user message
    const newMessages = [...messages, { id: Date.now(), role: 'user', text: input }]
    setMessages(newMessages)
    setInput('')
    
    // Simulate AI response
    setTimeout(() => {
      setMessages([...newMessages, { 
        id: Date.now() + 1, 
        role: 'bot', 
        text: 'Berdasarkan data saat ini, kamera yang paling ramai adalah Simpang Buah Batu dengan tingkat kepadatan "Sedang".' 
      }])
    }, 1000)
  }

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl">
          <MessageSquare className="h-5 w-5 text-primary" /> Traffic Assistant
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-[400px] p-4">
          <div className="space-y-4">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${msg.role === 'bot' ? 'bg-primary/20 text-primary' : 'bg-muted text-muted-foreground'}`}>
                  {msg.role === 'bot' ? <Bot size={18} /> : <User size={18} />}
                </div>
                <div className={`rounded-lg px-4 py-2 max-w-[80%] text-sm ${msg.role === 'bot' ? 'bg-muted/50' : 'bg-primary text-primary-foreground'}`}>
                  {msg.text}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
      <CardFooter className="p-4 border-t border-border">
        <form 
          className="flex w-full gap-2"
          onSubmit={(e) => { e.preventDefault(); handleSend(); }}
        >
          <Input 
            placeholder="Tanyakan jam berapa paling padat hari ini..." 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-1 bg-background/50"
          />
          <Button type="submit" disabled={!input.trim()}>
            <Send className="w-4 h-4 mr-2" /> Kirim
          </Button>
        </form>
      </CardFooter>
    </Card>
  )
}
