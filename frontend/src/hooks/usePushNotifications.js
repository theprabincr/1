import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// Convert base64 to Uint8Array for VAPID key
function urlBase64ToUint8Array(base64String) {
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding)
    .replace(/-/g, '+')
    .replace(/_/g, '/');
  
  const rawData = window.atob(base64);
  const outputArray = new Uint8Array(rawData.length);
  
  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i);
  }
  return outputArray;
}

export const usePushNotifications = () => {
  const [isSupported, setIsSupported] = useState(false);
  const [isSubscribed, setIsSubscribed] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [permission, setPermission] = useState('default');
  const [error, setError] = useState(null);

  // Check if push notifications are supported
  useEffect(() => {
    const supported = 'serviceWorker' in navigator && 'PushManager' in window;
    setIsSupported(supported);
    
    if (supported) {
      setPermission(Notification.permission);
    }
    
    checkSubscription();
  }, []);

  // Check current subscription status
  const checkSubscription = useCallback(async () => {
    try {
      if (!('serviceWorker' in navigator)) {
        setIsLoading(false);
        return;
      }

      const registration = await navigator.serviceWorker.ready;
      const subscription = await registration.pushManager.getSubscription();
      
      setIsSubscribed(!!subscription);
      setIsLoading(false);
    } catch (err) {
      console.error('Error checking subscription:', err);
      setIsLoading(false);
    }
  }, []);

  // Subscribe to push notifications
  const subscribe = useCallback(async () => {
    setError(null);
    setIsLoading(true);

    try {
      // Request permission
      const permissionResult = await Notification.requestPermission();
      setPermission(permissionResult);

      if (permissionResult !== 'granted') {
        setError('Notification permission denied');
        setIsLoading(false);
        return false;
      }

      // Register service worker
      const registration = await navigator.serviceWorker.register('/sw.js');
      await navigator.serviceWorker.ready;

      // Get VAPID public key from server
      const { data } = await axios.get(`${API}/push/vapid-public-key`);
      const vapidPublicKey = urlBase64ToUint8Array(data.publicKey);

      // Subscribe to push manager
      const subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: vapidPublicKey
      });

      // Send subscription to server
      await axios.post(`${API}/push/subscribe`, subscription.toJSON());

      setIsSubscribed(true);
      setIsLoading(false);
      return true;
    } catch (err) {
      console.error('Error subscribing to push:', err);
      setError(err.message || 'Failed to subscribe');
      setIsLoading(false);
      return false;
    }
  }, []);

  // Unsubscribe from push notifications
  const unsubscribe = useCallback(async () => {
    setError(null);
    setIsLoading(true);

    try {
      const registration = await navigator.serviceWorker.ready;
      const subscription = await registration.pushManager.getSubscription();

      if (subscription) {
        // Unsubscribe from push manager
        await subscription.unsubscribe();
        
        // Remove from server
        await axios.post(`${API}/push/unsubscribe`, subscription.toJSON());
      }

      setIsSubscribed(false);
      setIsLoading(false);
      return true;
    } catch (err) {
      console.error('Error unsubscribing from push:', err);
      setError(err.message || 'Failed to unsubscribe');
      setIsLoading(false);
      return false;
    }
  }, []);

  // Test push notification
  const testPush = useCallback(async () => {
    try {
      await axios.post(`${API}/push/test`);
      return true;
    } catch (err) {
      console.error('Error sending test push:', err);
      setError(err.message || 'Failed to send test');
      return false;
    }
  }, []);

  return {
    isSupported,
    isSubscribed,
    isLoading,
    permission,
    error,
    subscribe,
    unsubscribe,
    testPush,
    checkSubscription
  };
};

export default usePushNotifications;
