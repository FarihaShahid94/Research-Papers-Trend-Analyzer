"use client"
 
import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Search } from 'lucide-react';
 
// Import the new async API function
import { getKeywordFrequency } from '@/lib/api'; // Adjust path if needed
 
interface KeywordFrequencyLookupProps {
  selectedCategory: string | null;
}
 
export default function KeywordFrequencyLookup({ selectedCategory }: KeywordFrequencyLookupProps) {
  const [keyword, setKeyword] = useState('');
  const [frequency, setFrequency] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [searchedTerm, setSearchedTerm] = useState('');
 
  const handleLookup = async () => {
    if (!keyword.trim() || !selectedCategory) {
      setFrequency(null);
      setSearchedTerm(keyword);
      return;
    }
    setLoading(true);
    setSearchedTerm(keyword);
    try {
      const freq = await getKeywordFrequency(selectedCategory, keyword);
      setFrequency(freq);
    } catch (error) {
      console.error('Error fetching frequency:', error);
      setFrequency(null);
    } finally {
      setLoading(false);
    }
  };
 
  return (
    <Card className="shadow-lg rounded-lg">
      <CardHeader>
        <CardTitle className="text-lg flex items-center">
          <Search className="mr-2 h-5 w-5 text-primary" />
          Keyword Frequency
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex space-x-2">
          <Input
            type="text"
            placeholder="Enter keyword (e.g., Transformer)"
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleLookup()}
            disabled={!selectedCategory}
          />
          <Button onClick={handleLookup} disabled={loading || !selectedCategory}>
            {loading ? 'Searching...' : 'Lookup'}
          </Button>
        </div>
        {!selectedCategory && (
          <p className="text-sm text-muted-foreground">Please select a category first.</p>
        )}
        {searchedTerm && !loading && (
          <div>
            {frequency !== null ? (
              <p>The keyword "<strong>{searchedTerm}</strong>" appeared <strong>{frequency.toLocaleString()}</strong> times in category "<strong>{selectedCategory}</strong>".</p>
            ) : (
              <p>No data found for "<strong>{searchedTerm}</strong>" in {selectedCategory === "" ? "all categories" : "category"} "<strong>{selectedCategory === "" ? "" : selectedCategory}</strong>".</p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
 