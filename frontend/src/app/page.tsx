"use client";

import ChatWidget from "@/components/ChatWidget";

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-50 flex items-center justify-center p-8">
      <div className="text-center max-w-2xl">
        <h1 className="text-3xl font-bold text-gray-800 mb-4">
          Chat Widget Test Environment
        </h1>
        <p className="text-gray-600 mb-8">
          This is a blank canvas. The chat widget is running in the bottom right corner.
          You can copy the <code>src/components/ChatWidget.tsx</code> file into any of your other React/Next.js projects to integrate it!
        </p>
      </div>

      {/* The AI Assistant Widget (Standalone) */}
      <ChatWidget />
      
    </main>
  );
}
