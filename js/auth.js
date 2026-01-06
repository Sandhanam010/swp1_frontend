

// Firebase SDK imports
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import {
  getAuth,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  signOut
} from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";

// 🔑 Firebase config
const firebaseConfig = {
  apiKey: "AIzaSyDqnPOJLah97QBRORiz_iY6PeSox1P9wOM",
  authDomain: "swp1-caa02.firebaseapp.com",
  projectId: "swp1-caa02",
  appId: "1:585073596320:web:02fabd0d3bb6328edab6c5"
};

// Initialize Firebase ONCE
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// -------------------- LOGIN --------------------
export async function login(email, password) {
  const cred = await signInWithEmailAndPassword(auth, email, password);
  return cred.user;
}

// -------------------- SIGNUP --------------------
export async function signup(email, password) {
  return await createUserWithEmailAndPassword(auth, email, password);
}

// -------------------- LOGOUT --------------------
export async function logout() {
  await signOut(auth);
  window.location.href = "swp1_login.html";
}

// -------------------- AUTH GUARD --------------------
export function requireAuth(redirectTo = "swp1_login.html") {
  onAuthStateChanged(auth, (user) => {
    if (!user) {
      window.location.replace(redirectTo);
    }
  });
}

// -------------------- REDIRECT IF LOGGED IN --------------------
export function redirectIfLoggedIn(redirectTo = "swp1_home.html") {
  onAuthStateChanged(auth, (user) => {
    if (user) {
      window.location.replace(redirectTo);
    }
  });
}

// -------------------- GET CURRENT USER --------------------
export function getCurrentUser(callback) {
  onAuthStateChanged(auth, (user) => {
    if (user) callback(user);
  });
}
