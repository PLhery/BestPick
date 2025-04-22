import React, { createContext, useContext, useReducer, useState } from 'react';
import { AppState, Photo } from '../types';
import { analyzeImage, groupSimilarPhotos, extractFeatures, prepareQualityEmbeddings } from '../utils/imageAnalysis';

type PhotoAction = 
  | { type: 'ADD_PHOTOS'; photos: Photo[] }
  | { type: 'TOGGLE_SELECT_PHOTO'; photoId: string }
  | { type: 'SELECT_ALL_IN_GROUP'; groupId: string }
  | { type: 'DESELECT_ALL_IN_GROUP'; groupId: string }
  | { type: 'SELECT_ALL' }
  | { type: 'DESELECT_ALL' }
  | { type: 'UPDATE_GROUPS' }
  | { type: 'UNDO' }
  | { type: 'REDO' };

const initialState: AppState = {
  photos: [],
  groups: [],
  selectedPhotos: [],
  uniquePhotos: [],
  history: [],
  currentHistoryIndex: -1,
};

function reducer(state: AppState, action: PhotoAction): AppState {
  let newState: AppState;

  switch (action.type) {
    case 'ADD_PHOTOS': {
      const newState: AppState = {
        ...state,
        photos: [...state.photos, ...action.photos],
      };
      // After adding photos, regroup all photos
      const { groups, uniquePhotos } = groupSimilarPhotos(newState.photos);
      newState.groups = groups;
      newState.uniquePhotos = uniquePhotos;
      
      // Auto-select unique photos and best photos from groups
      const autoSelectedPhotos = [
        ...uniquePhotos.map(photo => photo.id),
        ...groups.map(group => group.photos[0].id)
      ];
      
      newState.selectedPhotos = autoSelectedPhotos;
      newState.photos = newState.photos.map(photo => ({
        ...photo,
        selected: autoSelectedPhotos.includes(photo.id)
      }));
      
      // Add initial state to history if it's the first addition
      if (state.photos.length === 0 && newState.photos.length > 0) {
        newState.history = [{ selectedPhotos: newState.selectedPhotos, timestamp: new Date() }];
        newState.currentHistoryIndex = 0;
      } else if (newState.photos.length > state.photos.length) {
        // Only add history if photos were actually added
        newState.history = [
          ...state.history.slice(0, state.currentHistoryIndex + 1),
          { selectedPhotos: newState.selectedPhotos, timestamp: new Date() }
        ];
        newState.currentHistoryIndex = state.currentHistoryIndex + 1;
      }

      return newState;
    }

    case 'TOGGLE_SELECT_PHOTO': {
      const selectedIndex = state.selectedPhotos.indexOf(action.photoId);
      const newSelectedPhotos = [...state.selectedPhotos];
      
      if (selectedIndex === -1) {
        newSelectedPhotos.push(action.photoId);
      } else {
        newSelectedPhotos.splice(selectedIndex, 1);
      }

      newState = {
        ...state,
        selectedPhotos: newSelectedPhotos,
        history: [
          ...state.history.slice(0, state.currentHistoryIndex + 1),
          { selectedPhotos: newSelectedPhotos, timestamp: new Date() }
        ],
        currentHistoryIndex: state.currentHistoryIndex + 1,
      };
      
      // Update the selected state in photos array
      newState.photos = state.photos.map(photo => 
        photo.id === action.photoId 
          ? { ...photo, selected: !photo.selected } 
          : photo
      );
      
      return newState;
    }

    case 'SELECT_ALL_IN_GROUP': {
      const group = state.groups.find(g => g.id === action.groupId);
      if (!group) return state;
      
      const photoIds = group.photos.map(photo => photo.id);
      const newSelectedPhotosSet = new Set([...state.selectedPhotos, ...photoIds]);
      const newSelectedPhotos = Array.from(newSelectedPhotosSet);
      
      // Check if selection actually changed before updating history
      if (newSelectedPhotos.length === state.selectedPhotos.length && newSelectedPhotos.every(id => state.selectedPhotos.includes(id))) {
        return state; // No change
      }

      newState = {
        ...state,
        selectedPhotos: newSelectedPhotos,
        history: [
          ...state.history.slice(0, state.currentHistoryIndex + 1),
          { selectedPhotos: newSelectedPhotos, timestamp: new Date() }
        ],
        currentHistoryIndex: state.currentHistoryIndex + 1,
      };
      
      // Update the selected state in photos array
      newState.photos = state.photos.map(photo => 
        photoIds.includes(photo.id) 
          ? { ...photo, selected: true } 
          : photo
      );
      
      return newState;
    }

    case 'DESELECT_ALL_IN_GROUP': {
      const group = state.groups.find(g => g.id === action.groupId);
      if (!group) return state;
      
      const photoIds = group.photos.map(photo => photo.id);
      const initialSelectedCount = state.selectedPhotos.length;
      const newSelectedPhotos = state.selectedPhotos.filter(id => !photoIds.includes(id));

      // Check if selection actually changed before updating history
      if (newSelectedPhotos.length === initialSelectedCount) {
        return state; // No change
      }

      newState = {
        ...state,
        selectedPhotos: newSelectedPhotos,
        history: [
          ...state.history.slice(0, state.currentHistoryIndex + 1),
          { selectedPhotos: newSelectedPhotos, timestamp: new Date() }
        ],
        currentHistoryIndex: state.currentHistoryIndex + 1,
      };
      
      // Update the selected state in photos array
      newState.photos = state.photos.map(photo => 
        photoIds.includes(photo.id) 
          ? { ...photo, selected: false } 
          : photo
      );
      
      return newState;
    }

    case 'SELECT_ALL': {
      const allPhotoIds = state.photos.map(photo => photo.id);
      
      // Check if selection actually changed before updating history
      if (allPhotoIds.length === state.selectedPhotos.length && allPhotoIds.every(id => state.selectedPhotos.includes(id))) {
        return state; // No change
      }

      newState = {
        ...state,
        selectedPhotos: allPhotoIds,
        history: [
          ...state.history.slice(0, state.currentHistoryIndex + 1),
          { selectedPhotos: allPhotoIds, timestamp: new Date() }
        ],
        currentHistoryIndex: state.currentHistoryIndex + 1,
      };
      
      // Update the selected state in photos array
      newState.photos = state.photos.map(photo => ({ ...photo, selected: true }));
      
      return newState;
    }

    case 'DESELECT_ALL': {
      // Check if selection actually changed before updating history
      if (state.selectedPhotos.length === 0) {
        return state; // No change
      }
      
      newState = {
        ...state,
        selectedPhotos: [],
        history: [
          ...state.history.slice(0, state.currentHistoryIndex + 1),
          { selectedPhotos: [], timestamp: new Date() }
        ],
        currentHistoryIndex: state.currentHistoryIndex + 1,
      };
      
      // Update the selected state in photos array
      newState.photos = state.photos.map(photo => ({ ...photo, selected: false }));
      
      return newState;
    }

    case 'UPDATE_GROUPS': {
      const { groups, uniquePhotos } = groupSimilarPhotos(state.photos);
      // Avoid unnecessary state update if groups/unique haven't changed
      if (
        JSON.stringify(groups) === JSON.stringify(state.groups) &&
        JSON.stringify(uniquePhotos) === JSON.stringify(state.uniquePhotos)
      ) {
        return state;
      }
      return {
        ...state,
        groups,
        uniquePhotos
      };
    }

    case 'UNDO': {
      if (state.currentHistoryIndex <= 0) return state;
      
      const historyState = state.history[state.currentHistoryIndex - 1];
      
      newState = {
        ...state,
        selectedPhotos: historyState.selectedPhotos,
        currentHistoryIndex: state.currentHistoryIndex - 1,
      };
      
      // Update the selected state in photos array
      newState.photos = state.photos.map(photo => ({
        ...photo,
        selected: historyState.selectedPhotos.includes(photo.id)
      }));
      
      return newState;
    }

    case 'REDO': {
      if (state.currentHistoryIndex >= state.history.length - 1) return state;
      
      const historyState = state.history[state.currentHistoryIndex + 1];
      
      newState = {
        ...state,
        selectedPhotos: historyState.selectedPhotos,
        currentHistoryIndex: state.currentHistoryIndex + 1,
      };
      
      // Update the selected state in photos array
      newState.photos = state.photos.map(photo => ({
        ...photo,
        selected: historyState.selectedPhotos.includes(photo.id)
      }));
      
      return newState;
    }

    default:
      return state;
  }
}

interface PhotoContextType {
  state: AppState;
  isLoading: boolean;
  addPhotos: (files: File[]) => void;
  toggleSelectPhoto: (photoId: string) => void;
  selectAllInGroup: (groupId: string) => void;
  deselectAllInGroup: (groupId: string) => void;
  selectAll: () => void;
  deselectAll: () => void;
  undo: () => void;
  redo: () => void;
  downloadSelected: () => void;
  isSelected: (id: string) => boolean;
}

const PhotoContext = createContext<PhotoContextType | undefined>(undefined);

// Add the usePhotoContext hook definition and export it
export function usePhotoContext() {
  const context = useContext(PhotoContext);
  if (context === undefined) {
    throw new Error('usePhotoContext must be used within a PhotoProvider');
  }
  return context;
}

export function PhotoProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [isLoading, setIsLoading] = useState(false);

  const addPhotos = async (files: File[]) => {
    if (!files.length) return;

    setIsLoading(true);

    // Step 0: Analyze basic metadata and create initial Photo objects
    const photoPrepared: Photo[] = await Promise.all(
      files.map(async (file) => {
        // Filter out non-image files earlier? Or handle errors in analyzeImage
        const id = `${file.name}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
        const url = URL.createObjectURL(file);
        const thumbnailUrl = url; // Placeholder
        
        return {
          id,
          file,
          url,
          thumbnailUrl,
          name: file.name,
          size: file.size,
          type: file.type,
          dateCreated: new Date(file.lastModified),
          selected: false, // Initial selection happens in reducer
          // embedding will be added next
        };
      })
    );

    // Step 1: Extract features (embeddings) for each photo
    // TODO: Add progress reporting using the callback
    const photosWithEmbeddings: Photo[] = await Promise.all(
      photoPrepared.map(async (photo) => {
        try {
          const embedding = await extractFeatures(photo /*, progressCallback */);
          return { ...photo, embedding };
        } catch (error) {
          console.error(`Failed to extract features for ${photo.name}:`, error);
          // Return the photo without embedding if extraction fails
          return photo; 
        }
      })
    );

    const text_embeds = await prepareQualityEmbeddings();

    // Step 2: Analyze basic metadata and create initial Photo objects
    const photosWithMetadata: Photo[] = await Promise.all(
      photosWithEmbeddings.map(async (photo) => {
        const { quality, metadata } = await analyzeImage(photo.file, photo.url, photo.embedding!, text_embeds);
        
        return {
          ...photo,
          quality,
          metadata,
        };
      })
    );

    console.log('withMetadataReady', photosWithMetadata);
    
    dispatch({ type: 'ADD_PHOTOS', photos: photosWithMetadata });
    setIsLoading(false);
  };

  const toggleSelectPhoto = (photoId: string) => {
    dispatch({ type: 'TOGGLE_SELECT_PHOTO', photoId });
  };

  const selectAllInGroup = (groupId: string) => {
    dispatch({ type: 'SELECT_ALL_IN_GROUP', groupId });
  };

  const deselectAllInGroup = (groupId: string) => {
    dispatch({ type: 'DESELECT_ALL_IN_GROUP', groupId });
  };

  const selectAll = () => {
    dispatch({ type: 'SELECT_ALL' });
  };

  const deselectAll = () => {
    dispatch({ type: 'DESELECT_ALL' });
  };

  const undo = () => {
    dispatch({ type: 'UNDO' });
  };

  const redo = () => {
    dispatch({ type: 'REDO' });
  };

  const downloadSelected = () => {
    state.selectedPhotos.forEach(photoId => {
      const photo = state.photos.find(p => p.id === photoId);
      if (photo) {
        // Revoke object URL after download? Maybe better to use file directly if possible.
        const link = document.createElement('a');
        link.href = photo.url;
        link.download = photo.name;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        // Revoke the object URL to free up memory
        URL.revokeObjectURL(photo.url);
      }
    });
  };

  const isSelected = (id: string) => {
    return state.selectedPhotos.includes(id);
  }

  return (
    <PhotoContext.Provider
      value={{
        state,
        isLoading,
        addPhotos,
        toggleSelectPhoto,
        selectAllInGroup,
        deselectAllInGroup,
        selectAll,
        deselectAll,
        undo,
        redo,
        downloadSelected,
        isSelected
      }}
    >
      {children}
    </PhotoContext.Provider>
  );
}