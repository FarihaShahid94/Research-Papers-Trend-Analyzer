"use client";

import { BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Bar, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface Props {
  data: { category: string; count: number }[];
}

export default function TotalPapersPerCategoryChart({ data }: Props) {
  return (
    <Card className="shadow-lg rounded-lg">
  <CardHeader>
    <CardTitle>Number of Papers per Category</CardTitle>
  </CardHeader>
  <CardContent className="overflow-visible">
    <div className="w-full h-[500px] overflow-visible">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 20, left: 20, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="category"
            angle={-45}
            textAnchor="end"
            interval={0}
            height={150}
            tick={{ fontSize: 10 }} // Smaller font for X-axis
          />
          <YAxis tick={{ fontSize: 10 }} /> {/* Smaller font for Y-axis */}
          <Tooltip />
          <Bar dataKey="count" fill="hsl(var(--primary))" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  </CardContent>
</Card>

  );
}
