import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./app/App";
import { AppProviders } from "./app/providers";
import "./styles.css";

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("React root element was not found.");

createRoot(rootElement).render(
  <StrictMode>
    <AppProviders>
      <App />
    </AppProviders>
  </StrictMode>,
);
