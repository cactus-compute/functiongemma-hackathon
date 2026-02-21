import { BrowserRouter, Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage.jsx";
import CreateProfile from "./pages/CreateProfile.jsx";
import ViewProfile from "./pages/ViewProfile.jsx";
import MyNetwork from "./pages/MyNetwork.jsx";
import SmartQuery from "./pages/SmartQuery.jsx";
import VoiceNote from "./pages/VoiceNote.jsx";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/create" element={<CreateProfile />} />
        <Route path="/profile/:id" element={<ViewProfile />} />
        <Route path="/network" element={<MyNetwork />} />
        <Route path="/smart-query" element={<SmartQuery />} />
        <Route path="/voice-note" element={<VoiceNote />} />
      </Routes>
    </BrowserRouter>
  );
}
