"use client"

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Clock } from 'lucide-react';
import { getSentimentAtTime } from '@/lib/mock-data';

export default function SentimentAtTime() {
  const [time, setTime] = useState(''); // Expects format like "YYYY-MM"
  const [sentiment, setSentiment] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [searchedTime, setSearchedTime] = useState('');

  const handleLookup = async () => {
    if (!time.match(/^\d{4}-\d{2}$/)) { // Basic validation for YYYY-MM
        alert("Please enter time in YYYY-MM format (e.g., 2023-06).");
        return;
    }
    setLoading(true);
    setSearchedTime(time);
    await new Promise(resolve => setTimeout(resolve, 500)); // Simulate API call
    const sentimentValue = getSentimentAtTime(time);
    setSentiment(sentimentValue);
    setLoading(false);
  };

  const getSentimentEmoji = (score: number | null) => {
    if (score === null) return '';
    if (score > 0.66) return '😊';
    if (score > 0.33) return '😐';
    return '😟';
  };

  return (
    <Card className="shadow-lg rounded-lg">
      <CardHeader>
        <CardTitle className="text-lg flex items-center">
          <Clock className="mr-2 h-5 w-5 text-primary" />
          Sentiment at Specific Time
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex space-x-2">
          <Input
            type="text"
            placeholder="YYYY-MM (e.g., 2023-06)"
            value={time}
            onChange={(e) => setTime(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleLookup()}
          />
          <Button onClick={handleLookup} disabled={loading}>
            {loading ? 'Checking...' : 'Check'}
          </Button>
        </div>
        {searchedTime && !loading && (
          <div>
            {sentiment !== null ? (
              <p>Sentiment for <strong>{searchedTime}</strong>: <strong>{sentiment.toFixed(2)}</strong> {getSentimentEmoji(sentiment)}</p>
            ) : (
              <p>No sentiment data found for <strong>{searchedTime}</strong>.</p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
