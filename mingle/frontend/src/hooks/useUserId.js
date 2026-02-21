import { useState } from "react";
import { v4 as uuidv4 } from "uuid";

const KEY = "mingle_user_id";

export default function useUserId() {
  const [userId] = useState(() => {
    let id = localStorage.getItem(KEY);
    if (!id) {
      id = uuidv4();
      localStorage.setItem(KEY, id);
    }
    return id;
  });
  return userId;
}
