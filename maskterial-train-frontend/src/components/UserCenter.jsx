import { Menu, Text, Avatar, Group, Box, Button } from "@mantine/core";
import { IconSettings, IconLogout } from "@tabler/icons-react";
import styles from "./UserCenter.module.css";

export function UserCenter({ userId, onLogout }) {
  const getDisplayName = () => {
    if (userId === "test_user") {
      return "Guest User";
    }
    return userId;
  };

  const getAvatarColor = () => {
    if (userId === "test_user") {
      return "gray";
    }
    return "blue";
  };

  const getAvatarText = () => {
    if (userId === "test_user") {
      return "G";
    }
    return userId.charAt(0).toUpperCase();
  };

  return (
    <Menu shadow="md" width={200}>
      <Menu.Target>
        <Box className={styles.userCenter}>
          <Group gap="xs" wrap="nowrap">
            <Avatar color={getAvatarColor()} radius="xl" size="sm">
              {getAvatarText()}
            </Avatar>
            <Text size="sm" fw={500} className={styles.userName}>
              {getDisplayName()}
            </Text>
          </Group>
        </Box>
      </Menu.Target>

      <Menu.Dropdown>
        <Menu.Label>User Information</Menu.Label>
        <Menu.Item>
          <Text size="sm">
            {userId === "test_user" ? "Guest Mode" : `User: ${userId}`}
          </Text>
        </Menu.Item>
        {userId === "test_user" && (
          <Menu.Item>
            <Text size="xs" c="dimmed">
              Images are not saved in guest mode
            </Text>
          </Menu.Item>
        )}
        <Menu.Divider />
        <Menu.Label>Actions</Menu.Label>
        <Menu.Item 
          leftSection={<IconLogout size={14} />}
          onClick={onLogout}
        >
          Back to Home
        </Menu.Item>
        <Menu.Item 
          leftSection={<IconSettings size={14} />}
          disabled
        >
          Settings (Coming Soon)
        </Menu.Item>
      </Menu.Dropdown>
    </Menu>
  );
}

