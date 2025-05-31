'use client';

import { ResponsiveContainer, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Bar } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { mockCategories } from '@/lib/mock-data';


interface CoolWordCount {
  category: string;
  count: number;
}

interface Props {
  data: CoolWordCount[];
}

export default function CoolWordCountChart({ data }: Props) {
    const formattedData = data.map(entry => {
        const categoryObj = mockCategories.find(cat => cat.id === entry.category);
        return {
          ...entry,
          category: categoryObj?.name ?? entry.category, // fallback to key if name not found
        };
      });
      
  return (
    <Card className="shadow-lg rounded-lg">
      <CardHeader>
        <CardTitle>Cool Word Count per Category</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="w-full h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={formattedData} margin={{ top: 20, right: 30, left: 10, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" angle={-45} textAnchor="end" interval={0} height={100} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#82ca9d" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
