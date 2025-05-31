export default function AppFooter() {
  return (
    <footer className="bg-card border-t border-border text-card-foreground mt-12">
      <div className="container mx-auto px-4 py-6 text-center text-sm">
        <p>&copy; {new Date().getFullYear()} Arxiv Insights. Powered by Facts & Curiosity.</p>
        <div className="mt-2 space-x-4">
          <a href="#" className="hover:text-primary hover:underline">About</a>
          <a href="#" className="hover:text-primary hover:underline">GitHub</a>
          <a href="#" className="hover:text-primary hover:underline">Privacy Policy</a>
        </div>
      </div>
    </footer>
  );
}
