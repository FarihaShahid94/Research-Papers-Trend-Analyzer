import type { SystemStats } from '@/types';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Database, KeyRound, FileText, Activity } from 'lucide-react'; // Using Activity for sentiment points

interface SystemStatsPanelProps {
  stats: SystemStats;
}

export default function SystemStatsPanel({ stats }: SystemStatsPanelProps) {
  const statItems = [
    { label: 'Total Categories', value: stats.total_categories, icon: <Database className="h-6 w-6 text-accent" /> },
    { label: 'Unique Keywords Tracked', value: stats.total_unique_keywords, icon: <KeyRound className="h-6 w-6 text-accent" /> },
    { label: 'Total Papers Analyzed', value: stats.total_papers_counted, icon: <FileText className="h-6 w-6 text-accent" /> },
    { label: 'Average TF-IDF', value: stats.average_tfidf, icon: <Activity className="h-6 w-6 text-accent" /> },
  ];

  return (
    <Card className="shadow-lg rounded-lg">
      <CardHeader>
        <CardTitle className="text-lg flex items-center">
          <Database className="mr-2 h-5 w-5 text-primary" />
          System Statistics
        </CardTitle>
        <CardDescription>Overview of the data processed by Arxiv Insights.</CardDescription>
      </CardHeader>
      <CardContent className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {statItems.map(item => (
          <div key={item.label} className="p-4 bg-secondary/30 rounded-md flex items-start space-x-3">
            {item.icon}
            <div>
              <p className="text-sm text-muted-foreground">{item.label}</p>
              <p className="text-xl font-bold">{item.value.toLocaleString()}</p>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
