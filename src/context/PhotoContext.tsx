import React, { createContext, useContext, useReducer, useState } from 'react';
import { AppState, Photo, PhotoGroup } from '../types';
import { analyzeImage, groupSimilarPhotos, extractFeatures, prepareQualityEmbeddings } from '../utils/imageAnalysis';

type PhotoAction = 
  | { type: 'SET_PROCESSED_PHOTOS'; photos: Photo[]; groups: PhotoGroup[]; uniquePhotos: Photo[] }
  | { type: 'TOGGLE_SELECT_PHOTO'; photoId: string }
  | { type: 'SELECT_ALL_IN_GROUP'; groupId: string }
  | { type: 'DESELECT_ALL_IN_GROUP'; groupId: string }
  | { type: 'SELECT_ALL' }
  | { type: 'DESELECT_ALL' }
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
    case 'SET_PROCESSED_PHOTOS': {
      const { photos, groups, uniquePhotos } = action;

      const updatedStateBase: Omit<AppState, 'history' | 'currentHistoryIndex'> = {
        ...state,
        photos: [...state.photos, ...photos],
        groups,
        uniquePhotos,
        selectedPhotos: [],
      };

      const autoSelectedPhotos = [
        ...uniquePhotos.map(photo => photo.id),
        ...groups.map(group => group.photos[0].id)
      ];
      
      updatedStateBase.selectedPhotos = autoSelectedPhotos;
      updatedStateBase.photos = updatedStateBase.photos.map(photo => ({
        ...photo,
        selected: autoSelectedPhotos.includes(photo.id)
      }));
      
      let history = state.history;
      let currentHistoryIndex = state.currentHistoryIndex;

      const werePhotosAdded = photos.length > 0;
      if (state.photos.length === 0 && werePhotosAdded) {
        history = [{ selectedPhotos: updatedStateBase.selectedPhotos, timestamp: new Date() }];
        currentHistoryIndex = 0;
      } else if (werePhotosAdded) {
        history = [
          ...state.history.slice(0, state.currentHistoryIndex + 1),
          { selectedPhotos: updatedStateBase.selectedPhotos, timestamp: new Date() }
        ];
        currentHistoryIndex = state.currentHistoryIndex + 1;
      }

      const finalState: AppState = {
        ...updatedStateBase,
        history,
        currentHistoryIndex,
      };

      return finalState;
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
      
      if (newSelectedPhotos.length === state.selectedPhotos.length && newSelectedPhotos.every(id => state.selectedPhotos.includes(id))) {
        return state;
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

      if (newSelectedPhotos.length === initialSelectedCount) {
        return state;
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
      
      newState.photos = state.photos.map(photo => 
        photoIds.includes(photo.id) 
          ? { ...photo, selected: false } 
          : photo
      );
      
      return newState;
    }

    case 'SELECT_ALL': {
      const allPhotoIds = state.photos.map(photo => photo.id);
      
      if (allPhotoIds.length === state.selectedPhotos.length && allPhotoIds.every(id => state.selectedPhotos.includes(id))) {
        return state;
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
      
      newState.photos = state.photos.map(photo => ({ ...photo, selected: true }));
      
      return newState;
    }

    case 'DESELECT_ALL': {
      if (state.selectedPhotos.length === 0) {
        return state;
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
      
      newState.photos = state.photos.map(photo => ({ ...photo, selected: false }));
      
      return newState;
    }

    case 'UNDO': {
      if (state.currentHistoryIndex <= 0) return state;
      
      const historyState = state.history[state.currentHistoryIndex - 1];
      
      newState = {
        ...state,
        selectedPhotos: historyState.selectedPhotos,
        currentHistoryIndex: state.currentHistoryIndex - 1,
      };
      
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

    try {
      const photoPrepared: Omit<Photo, 'quality' | 'metadata' | 'embedding'>[] = files.map(file => {
        const id = `${file.name}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
        const url = URL.createObjectURL(file);
        const thumbnailUrl = url;

        return {
          id,
          file,
          url,
          thumbnailUrl,
          name: file.name,
          size: file.size,
          type: file.type,
          dateCreated: new Date(file.lastModified),
          selected: false,
        };
      });

      const photosWithEmbeddings: (Omit<Photo, 'quality' | 'metadata'> & { embedding?: number[] })[] = await Promise.all(
        photoPrepared.map(async (photo) => {
          try {
            const embedding = await extractFeatures(photo as Photo);
            return { ...photo, embedding };
          } catch (error) {
            console.error(`Failed to extract features for ${photo.name}:`, error);
            return { ...photo, embedding: undefined };
          }
        })
      );

      const text_embeds = await prepareQualityEmbeddings();

      const photosWithMetadata: Photo[] = await Promise.all(
        photosWithEmbeddings.map(async (photo) => {
          if (!photo.embedding) {
            console.warn(`Skipping quality/metadata analysis for ${photo.name} due to missing embedding.`);
            return {
              ...photo,
              quality: 0,
              metadata: { captureDate: photo.dateCreated },
              embedding: undefined
            } as Photo;
          }

          try {
            const { quality, metadata } = await analyzeImage(photo.file, photo.url, photo.embedding, text_embeds);
            return {
              ...photo,
              quality,
              metadata,
              embedding: photo.embedding
            };
          } catch (error) {
            console.error(`Failed to analyze image ${photo.name}:`, error);
            return {
              ...photo,
              quality: 0,
              metadata: { captureDate: photo.dateCreated },
              embedding: photo.embedding
            } as Photo;
          }
        })
      );


      const allPhotosToGroup = [...state.photos, ...photosWithMetadata];
      const { groups, uniquePhotos } = await groupSimilarPhotos(allPhotosToGroup);

      dispatch({
        type: 'SET_PROCESSED_PHOTOS',
        photos: photosWithMetadata,
        groups,
        uniquePhotos
      });

    } catch (error) {
      console.error("Error processing photos:", error);
    } finally {
      setIsLoading(false);
    }
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
        const link = document.createElement('a');
        link.href = photo.url;
        link.download = photo.name;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
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