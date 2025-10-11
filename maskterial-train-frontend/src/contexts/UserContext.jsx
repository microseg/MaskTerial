import { createContext, useContext, useState, useEffect } from "react";

const UserContext = createContext();

export function UserProvider({ children }) {
  // Initialize with test_user as default
  const [userId, setUserId] = useState("test_user");

  // Store userId in sessionStorage to persist across page reloads
  useEffect(() => {
    const storedUserId = sessionStorage.getItem("userId");
    if (storedUserId) {
      setUserId(storedUserId);
    } else {
      sessionStorage.setItem("userId", "test_user");
    }
  }, []);

  const updateUserId = (newUserId) => {
    setUserId(newUserId);
    sessionStorage.setItem("userId", newUserId);
  };

  return (
    <UserContext.Provider value={{ userId, updateUserId }}>
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

