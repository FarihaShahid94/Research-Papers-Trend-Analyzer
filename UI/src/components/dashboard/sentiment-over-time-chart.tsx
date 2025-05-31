"use client"

import type { SentimentDataPoint } from '@/types';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { ChartContainer, ChartTooltip, ChartTooltipContent, ChartLegend, ChartLegendContent } from '@/components/ui/chart';
import { LineChart as RechartsLineChart, CartesianGrid, XAxis, YAxis, Line, ResponsiveContainer, Legend } from 'recharts';
import { LineChart } from 'lucide-react'; // Icon

interface SentimentOverTimeChartProps {
  data: SentimentDataPoint[];
  categoryName?: string;
}

const chartConfig = {
  sentiment: {
    label: "Sentiment Score",
    color: "hsl(var(--accent))",
  },
} satisfies import("@/components/ui/chart").ChartConfig;

export default function SentimentOverTimeChart({ data, categoryName }: SentimentOverTimeChartProps) {
  return (
    <Card className="shadow-lg rounded-lg">
      <CardHeader>
        <CardTitle className="text-lg flex items-center">
          <LineChart className="mr-2 h-5 w-5 text-primary" />
          Sentiment Over Time {categoryName ? `for ${categoryName}` : ''}
        </CardTitle>
        <CardDescription>Visualizing how sentiment has evolved monthly.</CardDescription>
      </CardHeader>
      <CardContent>
        {data.length > 0 ? (
          <ChartContainer config={chartConfig} className="h-[300px] w-full">
            <RechartsLineChart data={data} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tickFormatter={(value) => value.substring(5)} />
              <YAxis domain={[0, 1]} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <ChartLegend content={<ChartLegendContent />} />
              <Line type="monotone" dataKey="sentiment" stroke="var(--color-sentiment)" strokeWidth={2} dot={false} />
            </RechartsLineChart>
          </ChartContainer>
        ) : (
          <p className="text-muted-foreground text-center py-10">No sentiment data available for the current selection.</p>
        )}
      </CardContent>
    </Card>
  );
}
