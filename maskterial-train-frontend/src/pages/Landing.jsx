import { Button, Stack, Title, Group } from "@mantine/core";
import { useUser } from "../contexts/UserContext";
import { useState } from "react";

export function Landing({ onEnter }) {
  const { updateUserId } = useUser();
  const [loading, setLoading] = useState(false);

  const enterAsGuest = () => {
    setLoading(true);
    updateUserId("guest_user");
    onEnter?.("guest");
  };

  const enterAsUser = () => {
    setLoading(true);
    // Keep current userId (default test_user) or let later pages handle login
    onEnter?.("user");
  };

  return (
    <Stack align="center" justify="center" style={{ height: "100vh" }} gap="lg">
      <Title order={1}>Lab Pencil</Title>
      <Group>
        <Button loading={loading} onClick={enterAsGuest}>Login as guest</Button>
        <Button variant="outline" loading={loading} onClick={enterAsUser}>Login as user</Button>
      </Group>
    </Stack>
  );
}


