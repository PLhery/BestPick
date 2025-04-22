import React from 'react';
import { FolderOpen } from 'lucide-react';

const EmptyState: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center px-4">
      <FolderOpen size={80} className="text-gray-600 mb-4" />
      <h2 className="text-2xl font-semibold text-white mb-2">
        No photos yet
      </h2>
      <p className="text-gray-400 max-w-md mb-8">
        Upload your photos to organize and declutter your collection. 
        We'll help you identify duplicate or similar photos and keep only the best ones.
      </p>
    </div>
  );
};

export default EmptyState;