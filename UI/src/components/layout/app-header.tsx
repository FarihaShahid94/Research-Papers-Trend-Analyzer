import { LayoutDashboard } from 'lucide-react';

export default function AppHeader() {
  return (
    <header className="bg-primary text-primary-foreground shadow-md">
      <div className="container mx-auto px-4 py-6 flex items-center space-x-3">
        <LayoutDashboard size={36} />
        <div>
          <h1 className="text-3xl font-bold">Arxiv Insights</h1>
          <p className="text-sm opacity-90">Explore research trends and insights from Arxiv data.</p>
        </div>
      </div>
    </header>
  );
}
