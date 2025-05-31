'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ArrowUpCircle, ArrowDownCircle, Sparkles } from 'lucide-react';
import { mockCategories } from '@/lib/mock-data'; // Adjust if coming from API or different path

interface TfIdfExtreme {
  category: string;
  score: number;
}

interface TfIdfExtremesProps {
  highest: TfIdfExtreme;
  lowest: TfIdfExtreme;
}

export default function TfIdfExtremes({ highest, lowest }: TfIdfExtremesProps) {
  const getCategoryName = (id: string) => {
    return mockCategories.find(cat => cat.id === id)?.name || id;
  };

  return (
    <Card className="shadow-lg rounded-lg">
      <CardHeader>
        <CardTitle className="text-lg flex items-center">
          <Sparkles className="mr-2 h-5 w-5 text-primary" />
          TF-IDF Extremes
        </CardTitle>
        <CardDescription>Categories with the most and least distinctive terms.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-md shadow">
          <div className="flex items-center text-blue-700 dark:text-blue-400">
            <ArrowUpCircle className="mr-2 h-5 w-5" />
            <h3 className="font-semibold">Highest Average TF-IDF</h3>
          </div>
          <p className="mt-1 text-sm">
            <strong>{getCategoryName(highest.category)}</strong> with a score of <strong>{highest.score.toFixed(2)}</strong>.
          </p>
        </div>
        <div className="p-3 bg-gray-100 dark:bg-gray-900/30 rounded-md shadow">
          <div className="flex items-center text-gray-700 dark:text-gray-400">
            <ArrowDownCircle className="mr-2 h-5 w-5" />
            <h3 className="font-semibold">Lowest Average TF-IDF</h3>
          </div>
          <p className="mt-1 text-sm">
            <strong>{getCategoryName(lowest.category)}</strong> with a score of <strong>{lowest.score.toFixed(2)}</strong>.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
