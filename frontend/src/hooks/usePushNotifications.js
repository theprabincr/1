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
  const [swRegistration, setSwRegistration] = useState(null);

  // Initialize - check support and register service worker
  useEffect(() => {
    const init = async () => {
      // Check browser support
      const supported = 'serviceWorker' in navigator && 
                       'PushManager' in window && 
                       'Notification' in window;
      
      setIsSupported(supported);
      console.log('[Push] Browser support:', supported);

      if (!supported) {
        setIsLoading(false);
        return;
      }

      // Get current permission status
      setPermission(Notification.permission);
      console.log('[Push] Current permission:', Notification.permission);

      try {
        // Register service worker early
        const registration = await navigator.serviceWorker.register('/sw.js', {
          scope: '/'
        });
        console.log('[Push] Service worker registered:', registration.scope);
        setSwRegistration(registration);

        // Wait for it to be ready
        await navigator.serviceWorker.ready;
        console.log('[Push] Service worker ready');

        // Check if already subscribed
        const subscription = await registration.pushManager.getSubscription();
        setIsSubscribed(!!subscription);
        console.log('[Push] Existing subscription:', !!subscription);

      } catch (err) {
        console.error('[Push] Init error:', err);
        setError(err.message);
      }

      setIsLoading(false);
    };

    init();
  }, []);

  // Subscribe to push notifications
  const subscribe = useCallback(async () => {
    console.log('[Push] Subscribe called');
    setError(null);
    setIsLoading(true);

    try {
      // Step 1: Request notification permission
      console.log('[Push] Requesting permission...');
      
      // Use a promise-based approach for better compatibility
      let permissionResult;
      
      if (Notification.permission === 'granted') {
        permissionResult = 'granted';
      } else if (Notification.permission === 'denied') {
        permissionResult = 'denied';
      } else {
        // Permission is 'default', need to ask
        permissionResult = await Notification.requestPermission();
      }
      
      console.log('[Push] Permission result:', permissionResult);
      setPermission(permissionResult);

      if (permissionResult !== 'granted') {
        setError('Please allow notifications when prompted');
        setIsLoading(false);
        return false;
      }

      // Step 2: Get or use existing service worker registration
      let registration = swRegistration;
      if (!registration) {
        console.log('[Push] Registering service worker...');
        registration = await navigator.serviceWorker.register('/sw.js');
        await navigator.serviceWorker.ready;
        setSwRegistration(registration);
      }
      console.log('[Push] Using registration:', registration.scope);

      // Step 3: Get VAPID public key from server
      console.log('[Push] Fetching VAPID key...');
      const { data } = await axios.get(`${API}/push/vapid-public-key`);
      
      if (!data.publicKey) {
        throw new Error('Server did not return VAPID key');
      }
      
      const vapidPublicKey = urlBase64ToUint8Array(data.publicKey);
      console.log('[Push] VAPID key received');

      // Step 4: Subscribe to push manager
      console.log('[Push] Subscribing to push manager...');
      const subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: vapidPublicKey
      });
      console.log('[Push] Subscription created:', subscription.endpoint.substring(0, 50) + '...');

      // Step 5: Send subscription to server
      console.log('[Push] Sending subscription to server...');
      await axios.post(`${API}/push/subscribe`, subscription.toJSON());
      console.log('[Push] Subscription saved to server');

      setIsSubscribed(true);
      setIsLoading(false);
      
      // Show a test notification to confirm it's working
      if (Notification.permission === 'granted') {
        new Notification('🎯 Ballzy Notifications Enabled!', {
          body: 'You will now receive alerts when new picks are available.',
          icon: '/logo192.png'
        });
      }
      
      return true;
    } catch (err) {
      console.error('[Push] Subscribe error:', err);
      setError(err.message || 'Failed to enable notifications');
      setIsLoading(false);
      return false;
    }
  }, [swRegistration]);

  // Unsubscribe from push notifications
  const unsubscribe = useCallback(async () => {
    console.log('[Push] Unsubscribe called');
    setError(null);
    setIsLoading(true);

    try {
      const registration = await navigator.serviceWorker.ready;
      const subscription = await registration.pushManager.getSubscription();

      if (subscription) {
        // Unsubscribe from push manager
        await subscription.unsubscribe();
        console.log('[Push] Unsubscribed from push manager');
        
        // Remove from server
        await axios.post(`${API}/push/unsubscribe`, subscription.toJSON());
        console.log('[Push] Removed from server');
      }

      setIsSubscribed(false);
      setIsLoading(false);
      return true;
    } catch (err) {
      console.error('[Push] Unsubscribe error:', err);
      setError(err.message || 'Failed to disable notifications');
      setIsLoading(false);
      return false;
    }
  }, []);

  // Test push notification
  const testPush = useCallback(async () => {
    console.log('[Push] Test push called');
    try {
      // First try a local notification
      if (Notification.permission === 'granted') {
        new Notification('🎯 Test from Ballzy', {
          body: 'Local notification test successful!',
          icon: '/logo192.png'
        });
      }
      
      // Then trigger server push
      await axios.post(`${API}/push/test`);
      console.log('[Push] Server test sent');
      return true;
    } catch (err) {
      console.error('[Push] Test error:', err);
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
    testPush
  };
};

export default usePushNotifications;
