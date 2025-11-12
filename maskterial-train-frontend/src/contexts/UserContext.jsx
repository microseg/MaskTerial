import { createContext, useContext, useState, useEffect } from "react";

const UserContext = createContext();

export function UserProvider({ children }) {
  // Initialize with test_user as default
  const [userId, setUserId] = useState("test_user");
  const [accessToken, setAccessToken] = useState(null);
  const [email, setEmail] = useState("");

  // Store userId and token in localStorage to persist across page reloads
  useEffect(() => {
    const storedUserId = localStorage.getItem("userId");
    const storedToken = localStorage.getItem("accessToken");
    const storedEmail = localStorage.getItem("email");
    
    if (storedUserId) {
      setUserId(storedUserId);
    } else {
      localStorage.setItem("userId", "test_user");
    }
    
    if (storedToken) {
      setAccessToken(storedToken);
    }
    
    if (storedEmail) {
      setEmail(storedEmail);
    }
  }, []);

  const updateUserId = (newUserId) => {
    setUserId(newUserId);
    localStorage.setItem("userId", newUserId);
  };
  
  const login = (userData) => {
    // userData should contain: { user_id, email, access_token }
    if (userData.user_id) {
      setUserId(userData.user_id);
      localStorage.setItem("userId", userData.user_id);
    }
    if (userData.email) {
      setEmail(userData.email);
      localStorage.setItem("email", userData.email);
    }
    if (userData.access_token) {
      setAccessToken(userData.access_token);
      localStorage.setItem("accessToken", userData.access_token);
    }
  };
  
  const logout = () => {
    setUserId("test_user");
    setAccessToken(null);
    setEmail("");
    localStorage.setItem("userId", "test_user");
    localStorage.removeItem("accessToken");
    localStorage.removeItem("email");
  };

  return (
    <UserContext.Provider value={{ 
      userId, 
      accessToken, 
      email,
      updateUserId, 
      login,
      logout,
      isAuthenticated: !!accessToken
    }}>
      {children}
    </UserContext.Provider>
  );
}

export function useUser() {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error("useUser must be used within a UserProvider");
  }
  return context;
}

