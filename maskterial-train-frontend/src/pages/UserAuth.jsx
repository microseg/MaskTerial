import { useState } from "react";
import { Tabs, Group, TextInput, Button, Box, Stack, Title, Text } from "@mantine/core";
import axios from "axios";
import { useUser } from "../contexts/UserContext";

const REGISTER_URL = import.meta.env.VITE_AUTH_REGISTER_URL || "/api/auth/register";
const LOGIN_URL = import.meta.env.VITE_AUTH_LOGIN_URL || "/api/auth/login";

export function UserAuth({ onSuccess }) {
  const userContext = useUser();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [isError, setIsError] = useState(false);

  const signup = async () => {
    setMessage("");
    setIsError(false);
    try {
      const res = await axios.post(REGISTER_URL, { email, password });
      setMessage("Sign up successful");
      // Don't call onSuccess() - stay on the auth page
    } catch (e) {
      setIsError(true);
      setMessage(e?.response?.data?.detail || "Signup failed");
    }
  };

  const login = async () => {
    setMessage("");
    setIsError(false);
    try {
      const res = await axios.post(LOGIN_URL, { email, password });
      setMessage(res.data?.message || "Login success");
      
      // Save user data and token using the new login function
      userContext.login({
        user_id: res.data.user_id,
        email: res.data.email,
        access_token: res.data.access_token
      });
      
      onSuccess?.();
    } catch (e) {
      setIsError(true);
      setMessage(e?.response?.data?.detail || "Login failed");
    }
  };

  return (
    <Box style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <Box style={{ width: "min(560px, 92vw)" }}>
        <Stack gap="md" align="stretch">
          <Title order={2} ta="center">User</Title>
          <Tabs defaultValue="login" radius="md" keepMounted={false}>
            <Tabs.List grow>
              <Tabs.Tab value="login">Login</Tabs.Tab>
              <Tabs.Tab value="signup">Signup</Tabs.Tab>
            </Tabs.List>

            <Tabs.Panel value="login" pt="md">
              <Stack gap="sm">
                <TextInput label="User ID (email)" value={email} onChange={(e) => setEmail(e.currentTarget.value)} />
                <TextInput label="Password" type="password" value={password} onChange={(e) => setPassword(e.currentTarget.value)} />
                <Button onClick={login}>Login</Button>
              </Stack>
            </Tabs.Panel>

            <Tabs.Panel value="signup" pt="md">
              <Stack gap="sm">
                <TextInput label="User ID (email)" value={email} onChange={(e) => setEmail(e.currentTarget.value)} />
                <TextInput label="Password" type="password" value={password} onChange={(e) => setPassword(e.currentTarget.value)} />
                <Button variant="outline" onClick={signup}>Signup</Button>
              </Stack>
            </Tabs.Panel>
          </Tabs>
          {message && (
            <Text c={isError ? "red" : "green"} ta="center">{message}</Text>
          )}
        </Stack>
      </Box>
    </Box>
  );
}
