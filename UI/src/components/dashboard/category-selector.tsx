"use client"

import type { Category } from '@/types';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { LibraryBig } from 'lucide-react';

interface CategorySelectorProps {
  categories: Category[];
  selectedCategory: string | null;
  onSelectCategory: (categoryId: string | null) => void;
}

export default function CategorySelector({ categories, selectedCategory, onSelectCategory }: CategorySelectorProps) {
  return (
    <Card className="shadow-lg rounded-lg">
      <CardHeader className="pb-4">
        <CardTitle className="text-lg flex items-center">
          <LibraryBig className="mr-2 h-5 w-5 text-primary" />
          Select Research Category
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Select onValueChange={(value) => onSelectCategory(value === "all" ? null : value)} value={selectedCategory || "all"}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Filter by category..." />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Categories</SelectItem>
            {categories.map((category) => (
              <SelectItem key={category.id} value={category.id}>
                {category.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </CardContent>
    </Card>
  );
}
