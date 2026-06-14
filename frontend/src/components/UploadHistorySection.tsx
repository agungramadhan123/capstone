import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'
import { Button } from './ui/button'
import { History } from 'lucide-react'

const MOCK_HISTORY = [
  { id: '1', filename: 'rekaman_pagi.mp4', date: '2026-06-14 08:00', duration: '15:20', status: 'Analyzed' },
  { id: '2', filename: 'cctv_sore_bubat.mp4', date: '2026-06-13 17:30', duration: '45:00', status: 'Analyzed' },
  { id: '3', filename: 'sample_macet.mp4', date: '2026-06-13 10:15', duration: '10:00', status: 'Processing' },
]

export default function UploadHistorySection() {
  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl">
          <History className="h-5 w-5 text-primary" /> Riwayat Video
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="rounded-md border border-border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Nama File</TableHead>
                <TableHead>Tanggal</TableHead>
                <TableHead>Durasi</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {MOCK_HISTORY.map((item) => (
                <TableRow key={item.id}>
                  <TableCell className="font-medium">{item.filename}</TableCell>
                  <TableCell className="text-muted-foreground">{item.date}</TableCell>
                  <TableCell className="text-muted-foreground">{item.duration}</TableCell>
                  <TableCell>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${item.status === 'Analyzed' ? 'bg-green-500/10 text-green-500' : 'bg-yellow-500/10 text-yellow-500'}`}>
                      {item.status}
                    </span>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  )
}
