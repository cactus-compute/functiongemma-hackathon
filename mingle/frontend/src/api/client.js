import axios from "axios";

const api = axios.create({ baseURL: "/api" });

// ---- Profiles ----
export const createProfile = (data) => api.post("/profiles", data).then((r) => r.data);
export const getProfile = (id) => api.get(`/profiles/${id}`).then((r) => r.data);
export const getAllProfiles = () => api.get("/profiles").then((r) => r.data);
export const updateProfile = (id, data) => api.put(`/profiles/${id}`, data).then((r) => r.data);

// ---- QR ----
export const getQR = (profileId) => api.get(`/qr/${profileId}`).then((r) => r.data);

// ---- Network ----
export const getNetwork = (userId) =>
  api.get("/network", { params: { userId } }).then((r) => r.data);
export const saveContact = (userId, profileId) =>
  api.post("/network", { userId, profileId }).then((r) => r.data);
export const removeContact = (userId, profileId) =>
  api.delete(`/network/${profileId}`, { params: { userId } }).then((r) => r.data);

// ---- Outreach ----
export const rankContacts = (payload) =>
  api.post("/outreach/rank", payload).then((r) => r.data);
export const draftOutreach = (payload) =>
  api.post("/outreach/draft", payload).then((r) => r.data);
