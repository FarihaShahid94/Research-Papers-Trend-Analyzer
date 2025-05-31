"use client"

import type { PaperDataPoint, DateRangePickerProps } from '@/types';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { ChartContainer, ChartTooltip, ChartTooltipContent, ChartLegend, ChartLegendContent } from '@/components/ui/chart';
import { BarChart as RechartsBarChart, CartesianGrid, XAxis, YAxis, Bar, ResponsiveContainer } from 'recharts';
import { Newspaper, CalendarDays } from 'lucide-react';
import { DateRangePicker } from '@/components/ui/date-range-picker';
import type { DateRange } from 'react-day-picker';
import { format, parse } from 'date-fns';

interface PapersOverTimeChartProps {
  data: PaperDataPoint[];
  dateRange?: DateRange;
  onDateRangeChange: (dateRange: DateRange | undefined) => void;
  categoryName?: string;
}

const chartConfig = {
  papers: {
    label: "Papers Published",
    color: "hsl(var(--primary))",
  },
} satisfies import("@/components/ui/chart").ChartConfig;

export default function PapersOverTimeChart({ data, dateRange, onDateRangeChange, categoryName }: PapersOverTimeChartProps) {
  
  const filteredData = data.filter(item => {
    if (!dateRange || !dateRange.from) return true; // No filter applied or from date is missing
    const itemDate = parse(item.date, 'yyyy-MM', new Date());
    if (dateRange.to) {
      return itemDate >= dateRange.from && itemDate <= dateRange.to;
    }
    return itemDate >= dateRange.from;
  });

  return (
    <Card className="shadow-lg rounded-lg">
      <CardHeader>
        <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-4">
          <div>
            <CardTitle className="text-lg flex items-center">
              <Newspaper className="mr-2 h-5 w-5 text-primary" />
              Papers Published Over Time {categoryName ? `for ${categoryName}` : ''}
            </CardTitle>
            <CardDescription>Monthly publication volume. Use the picker to filter by date.</CardDescription>
          </div>
          <div className="w-full sm:w-auto min-w-[280px]">
             <DateRangePicker dateRange={dateRange} onDateRangeChange={onDateRangeChange} />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {filteredData.length > 0 ? (
          <ChartContainer config={chartConfig} className="h-[300px] w-full">
            <RechartsBarChart data={filteredData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date"
                tickFormatter={(value) => {
                  try {
                    const parsedDate = parse(value, 'yyyy-MM-dd HH:mm:ss', new Date());
                    return format(parsedDate, 'MMM d, HH:mm'); // Example: "May 30, 08:55"
                  } catch {
                    return value;
                  }
                }} 
              />
              <YAxis />
              <ChartTooltip content={<ChartTooltipContent />} />
              <ChartLegend content={<ChartLegendContent />} />
              <Bar dataKey="count" name="Papers" fill="var(--color-papers)" radius={[4, 4, 0, 0]} />
            </RechartsBarChart>
          </ChartContainer>
        ) : (
          <p className="text-muted-foreground text-center py-10">No publication data available for the selected range/category.</p>
        )}
      </CardContent>
    </Card>
  );
}
