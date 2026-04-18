import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  StatusBar,
  Dimensions,
  Platform,
  FlatList,
  Image,
  NativeModules,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import Constants from "expo-constants";

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get("window");

// ── Sequence Decoder (mirrors backend logic) ─────────────────
function collapseRepeats(seq) {
  if (!seq.length) return [];
  const out = [seq[0]];
  for (let i = 1; i < seq.length; i++) {
    if (seq[i] !== out[out.length - 1]) out.push(seq[i]);
  }
  return out;
}

function majorityVote(preds, windowSize = 5) {
  if (preds.length <= windowSize) {
    const counts = {};
    preds.forEach((p) => (counts[p] = (counts[p] || 0) + 1));
    const best = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0];
    return preds.map(() => best || "");
  }
  const half = Math.floor(windowSize / 2);
  return preds.map((_, i) => {
    const start = Math.max(0, i - half);
    const end = Math.min(preds.length, i + half + 1);
    const window = preds.slice(start, end);
    const counts = {};
    window.forEach((p) => (counts[p] = (counts[p] || 0) + 1));
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  });
}

function decodeSequence(rawPreds) {
  if (!rawPreds.length) return "";
  const smoothed = majorityVote(rawPreds);
  const collapsed = collapseRepeats(smoothed);
  return collapsed.join("");
}

// ── Constants & Assets ───────────────────────────────────────
const LABELS = "ABCDEFGHIKLMNOPQRSTUVWXY".split("");

const IMAGES = {
  A: require("./assets/practice/A.jpg"),
  B: require("./assets/practice/B.jpg"),
  C: require("./assets/practice/C.jpg"),
  D: require("./assets/practice/D.jpg"),
  E: require("./assets/practice/E.jpg"),
  F: require("./assets/practice/F.jpg"),
  G: require("./assets/practice/G.jpg"),
  H: require("./assets/practice/H.jpg"),
  I: require("./assets/practice/I.jpg"),
  K: require("./assets/practice/K.jpg"),
  L: require("./assets/practice/L.jpg"),
  M: require("./assets/practice/M.jpg"),
  N: require("./assets/practice/N.jpg"),
  O: require("./assets/practice/O.jpg"),
  P: require("./assets/practice/P.jpg"),
  Q: require("./assets/practice/Q.jpg"),
  R: require("./assets/practice/R.jpg"),
  S: require("./assets/practice/S.jpg"),
  T: require("./assets/practice/T.jpg"),
  U: require("./assets/practice/U.jpg"),
  V: require("./assets/practice/V.jpg"),
  W: require("./assets/practice/W.jpg"),
  X: require("./assets/practice/X.jpg"),
  Y: require("./assets/practice/Y.jpg"),
};

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();

  // Navigation State
  // "home", "free", "practice_list", "practice_detail"
  const [screen, setScreen] = useState("home");

  // Shared Inference State
  const [isRecording, setIsRecording] = useState(false);
  const [decodedText, setDecodedText] = useState("");
  const [currentLetter, setCurrentLetter] = useState("");
  const [currentConfidence, setCurrentConfidence] = useState(0);
  const [frameCount, setFrameCount] = useState(0);

  const [practiceLetter, setPracticeLetter] = useState(null);

  const predBuffer = useRef([]);
  const intervalRef = useRef(null);
  const cameraRef = useRef(null);
  const isProcessingRef = useRef(false);

  const runInference = useCallback(async () => {
    if (!cameraRef.current || isProcessingRef.current) return;

    isProcessingRef.current = true;
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.1,
        base64: true,
        skipProcessing: true,
        shutterSound: false
      });

      // Automatically detect laptop local IP from Expo bundler
      let IP = "10.7.27.226"; // Fallback from Metro logs
      try {
        if (NativeModules?.SourceCode?.scriptURL) {
          IP = NativeModules.SourceCode.scriptURL.split("://")[1].split(":")[0];
        } else if (Constants?.expoConfig?.hostUri) {
          IP = Constants.expoConfig.hostUri.split(":")[0];
        }
      } catch (err) { }

      const res = await fetch(`http://${IP}:5000/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: photo.base64 })
      });

      const data = await res.json();

      if (data.found && data.letter !== "?") {
        const letter = data.letter;
        const confidence = data.confidence;

        predBuffer.current.push(letter);
        setCurrentLetter(letter);
        setCurrentConfidence(confidence);
        setFrameCount((c) => c + 1);

        if (predBuffer.current.length % 5 === 0) {
          const decoded = decodeSequence(predBuffer.current);
          setDecodedText(decoded);
        }
      } else {
        setCurrentLetter("?");
        setCurrentConfidence(0);
      }
    } catch (e) {
      console.warn("API Error:", e?.message);
    } finally {
      isProcessingRef.current = false;
    }
  }, []);

  const startRecording = useCallback(() => {
    predBuffer.current = [];
    setDecodedText("");
    setCurrentLetter("");
    setCurrentConfidence(0);
    setFrameCount(0);
    setIsRecording(true);
    // Increased to 10 Hz
    intervalRef.current = setInterval(runInference, 100);
  }, [runInference]);

  const stopRecording = useCallback(() => {
    setIsRecording(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (predBuffer.current.length > 0) {
      const decoded = decodeSequence(predBuffer.current);
      setDecodedText(decoded);
    }
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  // Auto stop when leaving screens
  useEffect(() => {
    if (screen === "home" || screen === "practice_list") {
      stopRecording();
    }
  }, [screen, stopRecording]);


  // ── Render Helpers ──────────────────────────────────────────

  if (!permission) {
    return <View style={styles.center}><Text style={styles.loadingText}>Loading…</Text></View>;
  }

  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.permTitle}>📷 Camera Required</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // >> HOME SCREEN
  if (screen === "home") {
    return (
      <View style={styles.center}>
        <StatusBar barStyle="light-content" backgroundColor="#0f0f17" />
        <Text style={styles.homeTitle}>🤟 Sign Language</Text>
        <Text style={styles.homeSubtitle}>Please select a mode to start:</Text>

        <TouchableOpacity style={[styles.button, { marginTop: 40, width: 250 }]} onPress={() => setScreen("free")}>
          <Text style={styles.buttonText}>Free Mode</Text>
        </TouchableOpacity>

        <TouchableOpacity style={[styles.button, { marginTop: 20, width: 250, backgroundColor: "#ec4899" }]} onPress={() => setScreen("practice_list")}>
          <Text style={styles.buttonText}>Practice Mode</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // >> FREE MODE SCREEN
  if (screen === "free") {
    return (
      <View style={styles.container}>
        <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />
        {/* Camera is underneath everything */}
        <CameraView style={StyleSheet.absoluteFillObject} ref={cameraRef} facing="back" />

        {/* Overlay is outside of CameraView */}
        <View style={styles.cameraOverlay}>
          <View style={styles.headerRow}>
            <TouchableOpacity style={styles.backBtn} onPress={() => setScreen("home")}>
              <Text style={styles.backBtnText}>← Back</Text>
            </TouchableOpacity>
            <Text style={styles.headerTitle}>Free Mode</Text>
            <View style={{ width: 60 }} />
          </View>

          {isRecording && currentLetter && currentLetter !== "?" ? (
            <View style={styles.letterBadge}>
              <Text style={styles.letterText}>{currentLetter}</Text>
              <Text style={styles.letterConfidence}>{currentConfidence}%</Text>
            </View>
          ) : null}

          <View style={styles.decodedContainer}>
            <Text style={styles.decodedLabel}>Decoded Output</Text>
            <Text style={styles.decodedText}>
              {decodedText || (isRecording ? "Listening…" : "Tap Start to begin")}
            </Text>
          </View>

          <View style={styles.controls}>
            <TouchableOpacity
              style={[styles.playBtn, isRecording && styles.stopBtn]}
              onPress={isRecording ? stopRecording : startRecording}
            >
              <Text style={styles.buttonText}>{isRecording ? "⏹ Stop" : "▶ Start"}</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    );
  }

  // >> PRACTICE LIST SCREEN
  if (screen === "practice_list") {
    return (
      <View style={styles.listContainer}>
        <View style={styles.listHeaderRow}>
          <TouchableOpacity style={styles.backBtnDark} onPress={() => setScreen("home")}>
            <Text style={styles.backBtnTextDark}>← Back</Text>
          </TouchableOpacity>
          <Text style={styles.listHeaderTitle}>Select to Practice</Text>
        </View>

        <FlatList
          data={LABELS}
          keyExtractor={(item) => item}
          numColumns={3}
          contentContainerStyle={styles.gridContent}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={styles.gridItem}
              onPress={() => {
                setPracticeLetter(item);
                setScreen("practice_detail");
              }}
            >
              <Image source={IMAGES[item]} style={styles.gridImage} />
              <View style={styles.gridLabelContainer}>
                <Text style={styles.gridLabel}>{item}</Text>
              </View>
            </TouchableOpacity>
          )}
        />
      </View>
    );
  }

  // >> PRACTICE DETAIL SCREEN
  if (screen === "practice_detail") {
    const isMatched = isRecording && currentLetter === practiceLetter;

    return (
      <View style={styles.container}>
        {/* Top half: Target Image */}
        <View style={styles.practiceTop}>
          <Image source={IMAGES[practiceLetter]} style={styles.practiceImageLocal} />
          <TouchableOpacity style={styles.backBtnAbsolute} onPress={() => { stopRecording(); setScreen("practice_list"); }}>
            <Text style={styles.backBtnTextDark}>← Back</Text>
          </TouchableOpacity>
          <View style={styles.targetLabelBox}>
            <Text style={styles.targetLabelText}>Target: {practiceLetter}</Text>
          </View>
        </View>

        {/* Bottom half: Webcam Feed */}
        <View style={[styles.practiceBottom, isRecording && {
          borderColor: isMatched ? "#22c55e" : "#ef4444",
          borderWidth: 6
        }]}>
          <CameraView style={StyleSheet.absoluteFillObject} ref={cameraRef} facing="back" />

          {/* Overlay outside CameraView */}
          {isRecording ? (
            <>
              <View style={[styles.matchBadge, { backgroundColor: isMatched ? "#22c55e" : "#ef4444" }]}>
                <Text style={styles.matchLetter}>{currentLetter || "?"}</Text>
              </View>
              <View style={styles.confidenceBox}>
                <Text style={styles.confidenceRawText}>Conf: {currentConfidence}%</Text>
              </View>
            </>
          ) : (
            <View style={styles.cameraCenterOverlay}>
              <TouchableOpacity style={styles.playBtn} onPress={startRecording}>
                <Text style={styles.buttonText}>▶ Start Camera</Text>
              </TouchableOpacity>
            </View>
          )}

          {isRecording && (
            <TouchableOpacity style={styles.stopOverlayBtn} onPress={stopRecording}>
              <Text style={styles.buttonText}>⏹ Stop</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    );
  }

  return null;
}

// ── Global Styles ───────────────────────────────────────────
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  center: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#0f0f17" },
  loadingText: { color: "#fff", fontSize: 18 },

  homeTitle: { color: "#fff", fontSize: 32, fontWeight: "800", marginBottom: 10 },
  homeSubtitle: { color: "#a5b4fc", fontSize: 16 },

  button: { backgroundColor: "#6366f1", paddingVertical: 14, paddingHorizontal: 30, borderRadius: 12, alignItems: "center" },
  buttonText: { color: "#fff", fontSize: 18, fontWeight: "700" },

  cameraOverlay: { flex: 1, justifyContent: "space-between", zIndex: 10 },
  headerRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingTop: 50, paddingHorizontal: 20 },
  backBtn: { backgroundColor: "rgba(0,0,0,0.5)", padding: 8, borderRadius: 8 },
  backBtnText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  headerTitle: { color: "#fff", fontSize: 20, fontWeight: "700" },

  letterBadge: { position: "absolute", top: 120, right: 20, backgroundColor: "rgba(99,102,241,0.9)", borderRadius: 12, padding: 10, alignItems: "center" },
  letterText: { color: "#fff", fontSize: 40, fontWeight: "800" },
  letterConfidence: { color: "#c7d2fe", fontSize: 12, marginTop: 4, fontWeight: "600" },

  decodedContainer: { position: "absolute", bottom: 120, left: 20, right: 20, backgroundColor: "rgba(15,15,23,0.85)", borderRadius: 16, padding: 20, borderWidth: 1, borderColor: "rgba(99,102,241,0.4)" },
  decodedLabel: { color: "#a5b4fc", fontSize: 12, fontWeight: "700", textTransform: "uppercase", marginBottom: 8 },
  decodedText: { color: "#fff", fontSize: 22, fontWeight: "600" },

  controls: { alignItems: "center", paddingBottom: 40 },
  playBtn: { backgroundColor: "#6366f1", paddingVertical: 16, paddingHorizontal: 48, borderRadius: 14 },
  stopBtn: { backgroundColor: "#ef4444" },

  listContainer: { flex: 1, backgroundColor: "#1e1e2d" },
  listHeaderRow: { flexDirection: "row", alignItems: "center", paddingTop: 50, paddingBottom: 20, paddingHorizontal: 20, backgroundColor: "#151521" },
  backBtnDark: { padding: 8, backgroundColor: "#2b2b40", borderRadius: 8, marginRight: 15 },
  backBtnTextDark: { color: "#fff", fontSize: 16, fontWeight: "600" },
  listHeaderTitle: { color: "#fff", fontSize: 22, fontWeight: "700" },

  gridContent: { padding: 10 },
  gridItem: { flex: 1, margin: 8, backgroundColor: "#2b2b40", borderRadius: 12, overflow: "hidden", aspectRatio: 0.8 },
  gridImage: { width: "100%", height: "75%", resizeMode: "cover" },
  gridLabelContainer: { flex: 1, justifyContent: "center", alignItems: "center" },
  gridLabel: { color: "#fff", fontSize: 24, fontWeight: "800" },

  practiceTop: { flex: 1, backgroundColor: "#111" },
  practiceImageLocal: { width: "100%", height: "100%", resizeMode: "contain" },
  backBtnAbsolute: { position: "absolute", top: 50, left: 20, backgroundColor: "rgba(0,0,0,0.6)", padding: 10, borderRadius: 8 },
  targetLabelBox: { position: "absolute", bottom: 20, left: 20, backgroundColor: "rgba(0,0,0,0.8)", padding: 12, borderRadius: 8 },
  targetLabelText: { color: "#fff", fontSize: 24, fontWeight: "800" },

  practiceBottom: { flex: 1, backgroundColor: "#000", overflow: "hidden" },
  cameraHalf: { flex: 1 },
  cameraCenterOverlay: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "rgba(0,0,0,0.5)", zIndex: 10 },

  matchBadge: { position: "absolute", top: 20, left: 20, width: 60, height: 60, borderRadius: 30, justifyContent: "center", alignItems: "center", elevation: 5, zIndex: 10 },
  matchLetter: { color: "#fff", fontSize: 32, fontWeight: "800" },

  confidenceBox: { position: "absolute", top: 20, right: 20, backgroundColor: "rgba(0,0,0,0.7)", paddingHorizontal: 12, paddingVertical: 6, borderRadius: 8, zIndex: 10 },
  confidenceRawText: { color: "#fff", fontSize: 14, fontWeight: "600" },

  stopOverlayBtn: { position: "absolute", bottom: 30, alignSelf: "center", backgroundColor: "#ef4444", paddingVertical: 14, paddingHorizontal: 40, borderRadius: 12, zIndex: 10 }
});
