import { useState } from "react";
import { Button, TextInput, Tabs, Group } from "@mantine/core";
import axios from "axios";

const REGISTER_URL = import.meta.env.VITE_AUTH_REGISTER_URL;
const LOGIN_URL = import.meta.env.VITE_AUTH_LOGIN_URL;
const FORGOT_URL = import.meta.env.VITE_AUTH_FORGOT_URL;

export function AuthPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");

  const doRegister = async () => {
    setMessage("");
    try {
      const res = await axios.post(REGISTER_URL, { email, password });
      setMessage(res.data?.message || "Registered");
    } catch (e) {
      setMessage(e?.response?.data?.detail || "Register failed");
    }
  };

  const doLogin = async () => {
    setMessage("");
    try {
      const res = await axios.post(LOGIN_URL, { email, password });
      setMessage(res.data?.message || "Login successful");
    } catch (e) {
      setMessage(e?.response?.data?.detail || "Login failed");
    }
  };

  const doForgot = async () => {
    setMessage("");
    try {
      const res = await axios.post(FORGOT_URL, { email });
      setMessage(res.data?.message || "Password sent if email exists");
    } catch (e) {
      setMessage(e?.response?.data?.detail || "Request failed");
    }
  };

  return (
    <Tabs defaultValue="login" radius="md" keepMounted={false}>
      <Tabs.List grow>
        <Tabs.Tab value="login">Login</Tabs.Tab>
        <Tabs.Tab value="register">Register</Tabs.Tab>
        <Tabs.Tab value="forgot">Forgot Password</Tabs.Tab>
      </Tabs.List>

      <Tabs.Panel value="login" pt="md">
        <Group align="end">
          <TextInput label="Email" value={email} onChange={(e) => setEmail(e.currentTarget.value)} />
          <TextInput label="Password" type="password" value={password} onChange={(e) => setPassword(e.currentTarget.value)} />
          <Button onClick={doLogin}>Login</Button>
        </Group>
      </Tabs.Panel>

      <Tabs.Panel value="register" pt="md">
        <Group align="end">
          <TextInput label="Email" value={email} onChange={(e) => setEmail(e.currentTarget.value)} />
          <TextInput label="Password" type="password" value={password} onChange={(e) => setPassword(e.currentTarget.value)} />
          <Button variant="outline" onClick={doRegister}>Register</Button>
        </Group>
      </Tabs.Panel>

      <Tabs.Panel value="forgot" pt="md">
        <Group align="end">
          <TextInput label="Email" value={email} onChange={(e) => setEmail(e.currentTarget.value)} />
          <Button variant="light" onClick={doForgot}>Send Password</Button>
        </Group>
      </Tabs.Panel>

      {message && <div style={{ marginTop: 12 }}>{message}</div>}
    </Tabs>
  );
}


