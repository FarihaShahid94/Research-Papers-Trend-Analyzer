'use client';

import { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Sparkles } from 'lucide-react';
import axios from 'axios';
import clsx from 'clsx';

export default function CoolWordsCloud() {
  const [words, setWords] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchCoolWords() {
      try {
        const res = await axios.get('http://localhost:8000/all-coolword');
        setWords(res.data.coolwords || []);
      } catch (err) {
        console.error('Failed to load cool words:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchCoolWords();
  }, []);

  return (
    <Card className="shadow-lg rounded-lg">
      <CardHeader className="flex items-center space-x-2">
        <Sparkles className="text-primary w-5 h-5" />
        <CardTitle className="text-lg">Cool ML/AI Words</CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="text-muted-foreground">Loading...</div>
        ) : (
          <div className="max-h-[400px] overflow-y-auto text-sm leading-relaxed">
            <div className="flex flex-wrap gap-2">
              {words.map((word, i) => (
                <span
                  key={word + i}
                  className={clsx(
                    "px-2 py-1 rounded-md bg-primary/10 text-primary",
                    "hover:bg-primary/20 transition text-xs font-medium"
                  )}
                >
                  {word}
                </span>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
