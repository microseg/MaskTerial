import "./App.css";
import { Tabs, Group, Box } from "@mantine/core";
import { TrainModelPage } from "./pages/TrainModelPage";
import { AvailableModelsPage } from "./pages/AvailableModelsPage";
import { UploadModelPage } from "./pages/UploadModelPage";
import { TestInference } from "./pages/TestInference";
import { UserCenter } from "./components/UserCenter";
import { useUser } from "./contexts/UserContext";

function App() {
  const { userId } = useUser();

  return (
    <Box style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      <Tabs
        defaultValue="train"
        style={{ 
          flex: 1,
          display: "flex",
          flexDirection: "column"
        }}
      >
        {/* Top Navigation Bar with Tabs and User Center */}
        <Group 
          justify="space-between" 
          align="center" 
          wrap="nowrap"
          style={{ 
            height: "60px",
            minHeight: "60px",
            borderBottom: "1px solid #dee2e6",
            backgroundColor: "#f8f9fa",
            paddingRight: "20px",
            gap: "20px"
          }}
        >
          <Tabs.List 
            style={{ 
              border: "none",
              backgroundColor: "transparent",
              flex: 1
            }}
          >
            <Tabs.Tab value="train">Train Model</Tabs.Tab>
            <Tabs.Tab value="upload">Upload Trained Model</Tabs.Tab>
            <Tabs.Tab value="check">Check Available Models</Tabs.Tab>
            <Tabs.Tab value="test">Test Inference</Tabs.Tab>
          </Tabs.List>
          
          <UserCenter userId={userId} />
        </Group>

        {/* Content Area */}
        <Box style={{ flex: 1, overflow: "auto", backgroundColor: "white" }}>
          <Tabs.Panel value="train" style={{ height: "100%", padding: "20px" }}>
            <TrainModelPage />
          </Tabs.Panel>
          <Tabs.Panel value="upload" style={{ height: "100%", padding: "20px" }}>
            <UploadModelPage />
          </Tabs.Panel>
          <Tabs.Panel value="check" style={{ height: "100%", padding: "20px" }}>
            <AvailableModelsPage />
          </Tabs.Panel>
          <Tabs.Panel value="test" style={{ height: "100%", padding: "20px" }}>
            <TestInference />
          </Tabs.Panel>
        </Box>
      </Tabs>
    </Box>
  );
}

export default App;
