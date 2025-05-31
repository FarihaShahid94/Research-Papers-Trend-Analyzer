import type { Keyword } from '@/types';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Flame } from 'lucide-react';

interface TrendingKeywordsPanelProps {
  keywords: Keyword[];
  title?: string;
  icon?: React.ReactNode;
}

export default function TrendingKeywordsPanel({ 
  keywords, 
  title = "Trending Keywords", 
  icon = <Flame className="mr-2 h-5 w-5 text-primary" /> 
}: TrendingKeywordsPanelProps) {
  return (
    <Card className="shadow-lg rounded-lg">
      <CardHeader>
        <CardTitle className="text-lg flex items-center">
          {icon}
          {title}
        </CardTitle>
        <CardDescription>Top {keywords.length} most discussed topics currently.</CardDescription>
      </CardHeader>
      <CardContent>
        {keywords.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {keywords.map((keyword) => (
              <Badge key={keyword.text} variant="secondary" className="text-sm py-1 px-3">
                {keyword.text} ({keyword.frequency})
              </Badge>
            ))}
          </div>
        ) : (
          <p className="text-muted-foreground">No trending keywords to display for the current selection.</p>
        )}
      </CardContent>
    </Card>
  );
}
