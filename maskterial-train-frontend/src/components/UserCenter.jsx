import { Menu, Text, Avatar, Group, Box } from "@mantine/core";
import { IconSettings } from "@tabler/icons-react";
import styles from "./UserCenter.module.css";

export function UserCenter({ userId }) {
  return (
    <Menu shadow="md" width={200}>
      <Menu.Target>
        <Box className={styles.userCenter}>
          <Group gap="xs" wrap="nowrap">
            <Avatar color="blue" radius="xl" size="sm">
              {userId.charAt(0).toUpperCase()}
            </Avatar>
            <Text size="sm" fw={500} className={styles.userName}>
              {userId}
            </Text>
          </Group>
        </Box>
      </Menu.Target>

      <Menu.Dropdown>
        <Menu.Label>User Information</Menu.Label>
        <Menu.Item>
          <Text size="sm">User ID: {userId}</Text>
        </Menu.Item>
        <Menu.Divider />
        <Menu.Label>Settings</Menu.Label>
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

