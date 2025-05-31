import type { ForecastDataPoint } from '@/types';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { TrendingUp } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface ForecastPanelProps {
  predictions: ForecastDataPoint[];
  averagePrediction: number;
}

export default function ForecastPanel({ predictions, averagePrediction }: ForecastPanelProps) {
  return (
    <Card className="shadow-lg rounded-lg">
      <CardHeader>
        <CardTitle className="text-lg flex items-center">
          <TrendingUp className="mr-2 h-5 w-5 text-primary" />
          Publication Forecast
        </CardTitle>
        <CardDescription>Predicted paper volume for upcoming intervals.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <ul className="space-y-2">
          {predictions.map((prediction) => (
            <li key={prediction.interval} className="flex justify-between items-center p-2 bg-secondary/20 rounded-md">
              <span>{prediction.interval}:</span>
              <Badge variant="outline" className="font-semibold">{prediction.predictedVolume.toLocaleString()} papers</Badge>
            </li>
          ))}
        </ul>
        <div className="border-t pt-3 mt-3">
          <p className="text-sm text-muted-foreground">Average Predicted Volume (Next 5 Intervals):</p>
          <p className="text-xl font-bold text-primary">{averagePrediction.toLocaleString()} papers</p>
        </div>
      </CardContent>
    </Card>
  );
}
