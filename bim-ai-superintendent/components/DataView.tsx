import React, { useEffect, useState, useRef, useMemo } from 'react';
import { AnalysisResult } from '../types';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Line, Html } from '@react-three/drei';
import * as THREE from 'three';
import { Box, TrendingUp, CheckCircle, AlertTriangle, Ruler, MousePointer2 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { BimItem } from '../types';

// Fix for JSX.IntrinsicElements errors with React Three Fiber
// Augment both global and module-scoped JSX namespaces to handle different TypeScript configurations
declare global {
  namespace JSX {
    interface IntrinsicElements {
      mesh: any;
      group: any;
      points: any;
      bufferGeometry: any;
      bufferAttribute: any;
      pointsMaterial: any;
      sphereGeometry: any;
      meshBasicMaterial: any;
      planeGeometry: any;
      boxGeometry: any;
      meshStandardMaterial: any;
      ambientLight: any;
      pointLight: any;
      primitive: any;
    }
  }
}

declare module 'react' {
  namespace JSX {
    interface IntrinsicElements {
      mesh: any;
      group: any;
      points: any;
      bufferGeometry: any;
      bufferAttribute: any;
      pointsMaterial: any;
      sphereGeometry: any;
      meshBasicMaterial: any;
      planeGeometry: any;
      boxGeometry: any;
      meshStandardMaterial: any;
      ambientLight: any;
      pointLight: any;
      primitive: any;
    }
  }
}

// --- NUVEM GLOBAL (fundo cinza, mostra forma completa do scan) ---
// Permite ver pontos que não caíram dentro de nenhum OBB ("órfãos"),
// essencial pra diagnosticar visualmente se o alinhamento ficou certo.
const GlobalPointsCloud = ({ jsonFile }: { jsonFile: string | undefined }) => {
  const [points, setPoints] = React.useState<Float32Array | null>(null);

  useEffect(() => {
    if (!jsonFile) { setPoints(null); return; }
    const baseUrl = 'http://localhost:8081/outputs/';
    const finalUrl = jsonFile.startsWith('http') ? jsonFile : `${baseUrl}${jsonFile}`;
    fetch(finalUrl)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(d => {
        if (d.positions && Array.isArray(d.positions)) {
          setPoints(new Float32Array(d.positions));
        }
      })
      .catch(err => console.error("Erro ao carregar nuvem global:", err));
  }, [jsonFile]);

  if (!points) return null;
  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={points.length / 3} array={points} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial
        size={0.015}                    // bem pequeno, fica de "fundo"
        sizeAttenuation
        color={new THREE.Color(0.45, 0.5, 0.55)}  // cinza-azulado
        transparent
        opacity={0.55}                  // semitransparente pra não dominar
        depthWrite={false}              // OBBs/pts coloridos sempre por cima
      />
    </points>
  );
};

// --- TEXTURA DE PONTO CIRCULAR ---
// Sprite gerado uma vez no módulo (evita recriar a cada instância de ObjectPoints).
// Gradiente radial dá borda suave: centro opaco até 70% do raio, vai pra alpha=0 na borda.
// Combinado com alphaTest no material, recortamos os "cantos" do quad e ficamos com um disco.
const POINT_SPRITE = (() => {
  const size = 64;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d')!;
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0.0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.7, 'rgba(255,255,255,1)');
  grad.addColorStop(1.0, 'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(c);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
})();

// --- COMPONENTE DOS PONTOS ---
// Quando o backend devolve `colors` no JSON (PLY com RGB do scanner), o material
// usa `vertexColors` e a `color` passada vira só fallback de tinta única (modo status).
const ObjectPoints = ({ jsonFile, color }: { jsonFile: string; color: string }) => {
  const [points, setPoints] = React.useState<Float32Array | null>(null);
  const [colors, setColors] = React.useState<Float32Array | null>(null);

  useEffect(() => {
    if (!jsonFile) return;

    // Constrói URL correta: API retorna "session_id/arquivo.json"
    const baseUrl = 'http://localhost:8081/outputs/';
    const finalUrl = jsonFile.startsWith('http') ? jsonFile : `${baseUrl}${jsonFile}`;

    fetch(finalUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        return res.json();
      })
      .then((data) => {
        if (data.positions && Array.isArray(data.positions)) {
          setPoints(new Float32Array(data.positions));
        }
        // Backend manda colors como uint8 0-255 (compactação ~3-4× vs float).
        // Three.js pede color attribute em [0,1] float, então normalizamos.
        if (Array.isArray(data.colors) && data.colors.length === data.positions.length) {
          const u8 = data.colors as number[];
          const f32 = new Float32Array(u8.length);
          for (let i = 0; i < u8.length; i++) f32[i] = u8[i] / 255;
          setColors(f32);
        } else {
          setColors(null);
        }
      })
      .catch((err) => console.error("Erro ao carregar pontos:", err));
  }, [jsonFile]);

  if (!points) return null;

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={points.length / 3}
          array={points}
          itemSize={3}
        />
        {colors && (
          <bufferAttribute
            attach="attributes-color"
            count={colors.length / 3}
            array={colors}
            itemSize={3}
          />
        )}
      </bufferGeometry>
      <pointsMaterial
        size={0.03}                        // 0.05 → 0.03: menos "blob", mais sensação de scan
        sizeAttenuation
        map={POINT_SPRITE}                 // sprite radial vira disco
        alphaTest={0.5}                    // recorta os cantos do quad (sem blending)
        transparent={false}                // alphaTest sozinho basta — evita problemas de depth sort
        depthWrite                         // ponto opaco com depth correto
        // Quando temos cor por vértice (RGB do scanner) ignoramos `color` e usamos vertexColors.
        // Sem colors → usa a `color` única (fallback por status).
        vertexColors={!!colors}
        color={colors ? undefined : new THREE.Color(color)}
      />
    </points>
  );
};

// --- FERRAMENTA DE MEDIÇÃO ---
type AxisLock = 'x' | 'y' | 'z' | 'none';

const FixedScaleMarker: React.FC<{ position: THREE.Vector3; color: string }> = ({ position, color }) => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(({ camera }) => {
    if (meshRef.current) {
      const dist = camera.position.distanceTo(meshRef.current.position);
      const scale = dist * 0.015;
      meshRef.current.scale.set(scale, scale, scale);
    }
  });

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[1, 16, 16]} />
      <meshBasicMaterial color={color} depthTest={false} transparent opacity={0.9} />
    </mesh>
  );
};

// --- GRADE VERTICAL DINÂMICA ---
interface VerticalGridProps {
  position: THREE.Vector3;
  visible: boolean;
  gridCellSize?: number;
  height?: number;
}

const DynamicVerticalGrid: React.FC<VerticalGridProps> = ({
  position,
  visible,
  gridCellSize = 1,
  height = 10
}) => {
  // Snap para o centro da célula mais próxima da grade do chão
  // Math.round para pegar a célula mais próxima, não a anterior
  const snappedX = Math.round(position.x / gridCellSize) * gridCellSize;
  const snappedZ = Math.round(position.z / gridCellSize) * gridCellSize;
  const cellX = Math.round(position.x / gridCellSize);
  const cellZ = Math.round(position.z / gridCellSize);

  const gridWidth = gridCellSize * 2;
  const lineSpacing = 0.5;
  const numVerticalLines = Math.floor(gridWidth / lineSpacing) + 1;
  const numHorizontalLines = Math.floor(height / lineSpacing) + 1;

  if (!visible) return null;

  return (
    <group position={[snappedX, 0, snappedZ]}>
      {/* XY Plane Grid (Blue) */}
      <group>
        {Array.from({ length: numVerticalLines }, (_, i) => {
          const x = -gridWidth / 2 + i * lineSpacing;
          return (
            <Line
              key={`xv-${i}`}
              points={[[x, 0, 0], [x, height, 0]]}
              color={i === Math.floor(numVerticalLines / 2) ? "#60a5fa" : "#3b82f6"}
              lineWidth={i === Math.floor(numVerticalLines / 2) ? 2 : 1}
              transparent
              opacity={0.6}
            />
          );
        })}
        {Array.from({ length: numHorizontalLines }, (_, i) => {
          const y = i * lineSpacing;
          return (
            <Line
              key={`xh-${i}`}
              points={[[-gridWidth / 2, y, 0], [gridWidth / 2, y, 0]]}
              color={i % 2 === 0 ? "#60a5fa" : "#3b82f6"}
              lineWidth={i % 2 === 0 ? 2 : 1}
              transparent
              opacity={0.6}
            />
          );
        })}
      </group>

      {/* ZY Plane Grid (Green) */}
      <group>
        {Array.from({ length: numVerticalLines }, (_, i) => {
          const z = -gridWidth / 2 + i * lineSpacing;
          return (
            <Line
              key={`zv-${i}`}
              points={[[0, 0, z], [0, height, z]]}
              color={i === Math.floor(numVerticalLines / 2) ? "#4ade80" : "#22c55e"}
              lineWidth={i === Math.floor(numVerticalLines / 2) ? 2 : 1}
              transparent
              opacity={0.6}
            />
          );
        })}
        {Array.from({ length: numHorizontalLines }, (_, i) => {
          const y = i * lineSpacing;
          return (
            <Line
              key={`zh-${i}`}
              points={[[0, y, -gridWidth / 2], [0, y, gridWidth / 2]]}
              color={i % 2 === 0 ? "#4ade80" : "#22c55e"}
              lineWidth={i % 2 === 0 ? 2 : 1}
              transparent
              opacity={0.6}
            />
          );
        })}
      </group>

      {/* Central axis line (yellow) */}
      <Line points={[[0, 0, 0], [0, height, 0]]} color="#fbbf24" lineWidth={3} />

      {/* Height markers */}
      {Array.from({ length: Math.floor(height) + 1 }, (_, i) => (
        <group key={i} position={[0, i, 0]}>
          <Line points={[[-0.2, 0, 0], [0.2, 0, 0]]} color="#fbbf24" lineWidth={3} />
          <Line points={[[0, 0, -0.2], [0, 0, 0.2]]} color="#fbbf24" lineWidth={3} />
          <Html position={[0.5, 0, 0]} center>
            <div className="bg-yellow-500 text-black px-1.5 py-0.5 rounded text-[10px] font-bold shadow-lg">
              {i}m
            </div>
          </Html>
        </group>
      ))}

      {/* Quadrant label - fixo no canto superior direito */}
      <Html fullscreen style={{ pointerEvents: 'none' }}>
        <div className="absolute top-4 right-4">
          <div className="bg-slate-900/95 text-white px-4 py-2 rounded-lg text-sm font-mono border-2 border-cyan-500 shadow-2xl">
            <div className="flex items-center space-x-3">
              <span className="text-cyan-400 font-bold">📍 Quadrante:</span>
              <span className="text-red-400 font-bold">X: {cellX}</span>
              <span className="text-green-400 font-bold">Z: {cellZ}</span>
            </div>
          </div>
        </div>
      </Html>

      {/* Height indicator at top */}
      <Html position={[0, height + 0.8, 0]} center>
        <div className="bg-yellow-500 text-black px-3 py-1 rounded text-xs font-bold shadow-lg">
          ↕ Altura máx: {height}m
        </div>
      </Html>

      {/* Ground marker */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
        <planeGeometry args={[gridCellSize, gridCellSize]} />
        <meshBasicMaterial color="#06b6d4" transparent opacity={0.3} />
      </mesh>
    </group>
  );
};

const MeasureTool: React.FC = () => {
  const { camera, scene, gl } = useThree();
  const raycaster = useRef(new THREE.Raycaster()).current;

  const [measureModeEnabled, setMeasureModeEnabled] = useState(false);
  const [isMeasuring, setIsMeasuring] = useState(false);
  const [startPoint, setStartPoint] = useState<THREE.Vector3 | null>(null);
  const [endPoint, setEndPoint] = useState<THREE.Vector3 | null>(null);
  const [axisLock, setAxisLock] = useState<AxisLock>('none');

  // Vertical Grid State
  const [showVerticalGrid, setShowVerticalGrid] = useState(false);
  const [gridPosition, setGridPosition] = useState<THREE.Vector3>(new THREE.Vector3());
  const [gridAnchored, setGridAnchored] = useState(false); // true = grade fixa no ponto

  const getLineColor = () => {
    switch (axisLock) {
      case 'x': return '#ef4444';
      case 'y': return '#22c55e';
      case 'z': return '#3b82f6';
      default: return '#f97316';
    }
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();

      // ESPAÇO = Toggle do modo de medição
      if (key === ' ' || e.code === 'Space') {
        e.preventDefault();
        setMeasureModeEnabled(prev => !prev);
        // Limpa medição ao desativar
        if (measureModeEnabled) {
          setIsMeasuring(false);
          setStartPoint(null);
          setEndPoint(null);
          setAxisLock('none');
        }
        return;
      }

      // ESC = Cancela medição atual
      if (key === 'escape') {
        setIsMeasuring(false);
        setStartPoint(null);
        setEndPoint(null);
        setAxisLock('none');
        setShowVerticalGrid(false);
        return;
      }

      // G = Toggle grade vertical (funciona sempre)
      if (key === 'g') {
        console.log('G pressed! Toggling vertical grid mode');
        setShowVerticalGrid(prev => {
          if (prev) {
            // Desativando: limpa a ancora
            setGridAnchored(false);
          }
          return !prev;
        });
        return;
      }

      // Atalhos de travamento de eixo (só funciona se modo ativado)
      if (!measureModeEnabled) return;

      switch (key) {
        case 'w': case 'arrowup':
          console.log('🟢 Travando eixo Y (verde - vertical)');
          setAxisLock('y');
          break;
        case 'a': case 'arrowleft': case 'arrowright': case 'd':
          console.log('🔴 Travando eixo X (vermelho)');
          setAxisLock('x');
          break;
        case 's': case 'arrowdown': case 'q': case 'e':
          console.log('🔵 Travando eixo Z (azul)');
          setAxisLock('z');
          break;
        case 'r':
          console.log('🟠 Modo livre');
          setAxisLock('none');
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [measureModeEnabled]);

  const handlePointerDown = (e: any) => {
    if (!measureModeEnabled) return; // Só funciona se modo ativado
    if (e.button !== 0) return;
    e.stopPropagation();

    // Se a grade está ancorada, usa X,Z da grade mas Y de onde clicou
    if (gridAnchored && showVerticalGrid) {
      // X,Z fixos da grade, Y do ponto clicado
      const gridStartPoint = new THREE.Vector3(
        Math.round(gridPosition.x),
        e.point?.y || 0,  // Y vem do clique (onde começar a medir altura)
        Math.round(gridPosition.z)
      );
      setStartPoint(gridStartPoint);
      setEndPoint(gridStartPoint);
      // Auto-trava no eixo Y para medir altura
      setAxisLock('y');
      console.log('📏 Medindo altura a partir de Y:', gridStartPoint.y.toFixed(2));
    } else {
      setStartPoint(e.point);
      setEndPoint(e.point);
    }
    setIsMeasuring(true);
  };

  const handlePointerMove = (e: any) => {
    let currentPoint = e.point ? e.point.clone() : null;

    // Só atualiza posição da grade se NÂO estiver ancorada (preview antes de ancorar)
    if (showVerticalGrid && !gridAnchored && currentPoint) {
      setGridPosition(currentPoint.clone());
    }

    if (!measureModeEnabled || !isMeasuring || !startPoint || !currentPoint) return;

    // Usa o ponto que o React Three Fiber já dá (do plano invisível)

    // Aplica travamento de eixo DEPOIS de pegar o ponto
    if (axisLock === 'x') {
      // Trava X: só pode mover em X, Y e Z ficam fixos
      currentPoint.y = startPoint.y;
      currentPoint.z = startPoint.z;
    } else if (axisLock === 'y') {
      // Trava Y: só pode mover em Y (vertical), X e Z ficam fixos
      currentPoint.x = startPoint.x;
      currentPoint.z = startPoint.z;
    } else if (axisLock === 'z') {
      // Trava Z: só pode mover em Z, X e Y ficam fixos
      currentPoint.x = startPoint.x;
      currentPoint.y = startPoint.y;
    }
    // Se axisLock === 'none', usa currentPoint sem modificar (modo livre)

    setEndPoint(currentPoint);
  };

  const handlePointerUp = () => {
    if (measureModeEnabled && isMeasuring) {
      setIsMeasuring(false);
    }
  };

  // Botão direito ancora a grade vertical no ponto clicado
  const handleRightClick = (e: any) => {
    e.nativeEvent?.preventDefault?.();

    if (!showVerticalGrid) return;

    const point = e.point;
    if (!point) return;

    console.log('📌 Ancorando grade vertical em:', point.x.toFixed(2), point.z.toFixed(2));
    setGridPosition(point.clone());
    setGridAnchored(true);
  };

  const distance = useMemo(() => {
    if (startPoint && endPoint) {
      return startPoint.distanceTo(endPoint).toFixed(3);
    }
    return "0.00";
  }, [startPoint, endPoint]);

  const midPoint = useMemo(() => {
    if (startPoint && endPoint) {
      return new THREE.Vector3().addVectors(startPoint, endPoint).multiplyScalar(0.5);
    }
    return new THREE.Vector3();
  }, [startPoint, endPoint]);

  return (
    <>
      {/* Plano horizontal (chão) - para modo livre e eixo X/Z */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, startPoint?.y || 0, 0]}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onContextMenu={handleRightClick}
        visible={axisLock !== 'y'}
      >
        <planeGeometry args={[1000, 1000]} />
        <meshBasicMaterial transparent opacity={0} side={THREE.DoubleSide} />
      </mesh>

      {/* Plano vertical (parede frontal) - para eixo Y */}
      <mesh
        rotation={[0, 0, 0]}
        position={[startPoint?.x || 0, 0, startPoint?.z || 0]}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onContextMenu={handleRightClick}
        visible={axisLock === 'y'}
      >
        <planeGeometry args={[1000, 1000]} />
        <meshBasicMaterial transparent opacity={0} side={THREE.DoubleSide} />
      </mesh>

      {/* Grade Vertical Dinâmica */}
      <DynamicVerticalGrid
        position={gridPosition}
        visible={showVerticalGrid}
      />

      {startPoint && (
        <>
          <FixedScaleMarker position={startPoint} color={getLineColor()} />
          {endPoint && (
            <>
              <FixedScaleMarker position={endPoint} color={getLineColor()} />
              <Line
                points={[startPoint, endPoint]}
                color={getLineColor()}
                lineWidth={3}
                depthTest={false}
              />
              <Html position={midPoint} center>
                <div className={`text-white px-3 py-2 rounded-lg text-sm font-mono border-2 shadow-xl whitespace-nowrap z-50 pointer-events-none ${axisLock === 'x' ? 'bg-red-600 border-red-400' :
                  axisLock === 'y' ? 'bg-green-600 border-green-400' :
                    axisLock === 'z' ? 'bg-blue-600 border-blue-400' :
                      'bg-orange-600 border-orange-400'
                  }`}>
                  <div className="font-bold">{distance}m</div>
                  {axisLock !== 'none' && (
                    <div className="text-xs opacity-80">Eixo {axisLock.toUpperCase()} travado</div>
                  )}
                </div>
              </Html>
            </>
          )}
        </>
      )}

      <Html fullscreen style={{ pointerEvents: 'none' }}>
        <div className="absolute top-4 left-4 pointer-events-none">
          <div className={`backdrop-blur text-white p-3 rounded-lg border shadow-2xl max-w-xs text-xs transition-all ${measureModeEnabled
            ? 'bg-orange-900/90 border-orange-500'
            : 'bg-slate-900/80 border-slate-700'
            }`}>
            <div className="flex items-center space-x-2 mb-2 border-b border-current pb-2">
              <Ruler className={`w-4 h-4 ${measureModeEnabled ? 'text-orange-300' : 'text-slate-400'}`} />
              <span className="font-bold">
                {measureModeEnabled ? 'MODO MEDIÇÃO ATIVO' : 'Ferramenta de Medição'}
              </span>
            </div>
            <div className="space-y-1 text-[10px] text-slate-300">
              <div>• <span className="font-bold text-yellow-300">ESPAÇO</span> = Ativar/Desativar medição</div>
              {measureModeEnabled && (
                <>
                  <div>• Clique e arraste para medir</div>
                  <div>• <span className="text-red-400">A</span> = Eixo X (vermelho)</div>
                  <div>• <span className="text-green-400">W</span> = Eixo Y (verde)</div>
                  <div>• <span className="text-blue-400">S</span> = Eixo Z (azul)</div>
                  <div>• <span className="text-orange-400">R</span> = Livre</div>
                  <div className="text-slate-400 italic">ESC para limpar</div>
                </>
              )}
              <div className="mt-1 pt-1 border-t border-slate-600">
                <div><span className="text-cyan-300">G</span> = Modo Grade Vertical</div>
                {showVerticalGrid && (
                  <div className="text-cyan-400 text-[9px] mt-1">
                    • Botão direito = Fixar grade no ponto
                  </div>
                )}
              </div>
            </div>
            {showVerticalGrid && (
              <div className={`mt-2 text-center p-1 rounded border ${gridAnchored
                ? 'bg-green-900/50 border-green-500/30'
                : 'bg-cyan-900/50 border-cyan-500/30'
                }`}>
                <span className={`font-bold text-[10px] ${gridAnchored ? 'text-green-200' : 'text-cyan-200'
                  }`}>
                  {gridAnchored ? '📌 Grade Fixada' : '👆 Clique direito para fixar'}
                </span>
              </div>
            )}
            {isMeasuring && (
              <div className="mt-2 text-center bg-yellow-900/50 p-1 rounded border border-yellow-500/30">
                <span className="text-yellow-200 font-bold text-[10px] animate-pulse">Medindo...</span>
              </div>
            )}
          </div>
        </div>
      </Html>

      {/* Câmera livre quando NÃO está medindo ativamente */}
      <OrbitControls makeDefault enabled={!isMeasuring} />
    </>
  );
};

// --- COMPONENTE DE BOUNDING BOX ---

// --- HITBOX CLICÁVEL (seleção de objetos no 3D) ---
// Triângulos das 6 faces do OBB (2 triângulos por face = 12 triângulos)
const OBB_FACES = new Uint16Array([
  0,1,2,  0,2,3,    // bottom
  4,6,5,  4,7,6,    // top
  0,4,5,  0,5,1,    // side1
  1,5,6,  1,6,2,    // side2
  2,6,7,  2,7,3,    // side3
  3,7,4,  3,4,0,    // side4
]);

const SelectableHitbox: React.FC<{
  item: any;
  isSelected: boolean;
  onSelect: (guid: string) => void;
}> = ({ item, isSelected, onSelect }) => {
  const corners = item.obb_corners;

  // ----- Modo OBB (caixa orientada) -----
  if (corners && Array.isArray(corners) && corners.length === 8) {
    const { meshGeom, lineGeom } = React.useMemo(() => {
      const positions = new Float32Array(8 * 3);
      for (let i = 0; i < 8; i++) {
        positions[i * 3]     = corners[i][0];
        positions[i * 3 + 1] = corners[i][1];
        positions[i * 3 + 2] = corners[i][2];
      }
      const mesh = new THREE.BufferGeometry();
      mesh.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      mesh.setIndex(new THREE.BufferAttribute(OBB_FACES, 1));
      mesh.computeVertexNormals();

      const line = new THREE.BufferGeometry();
      line.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      line.setIndex(new THREE.BufferAttribute(OBB_EDGES, 1));
      return { meshGeom: mesh, lineGeom: line };
    }, [corners]);

    return (
      <group>
        <mesh
          geometry={meshGeom}
          onClick={(e) => { e.stopPropagation(); onSelect(item.guid); }}
        >
          <meshBasicMaterial visible={false} />
        </mesh>
        {isSelected && (
          <lineSegments geometry={lineGeom}>
            <lineBasicMaterial color="#00ffff" transparent opacity={0.9} />
          </lineSegments>
        )}
      </group>
    );
  }

  // ----- Fallback AABB -----
  const bbox = item.bbox_normalized || item.bbox;
  if (!bbox) return null;

  const { xmin, xmax, ymin, ymax, zmin, zmax } = bbox;
  const width = Math.abs(xmax - xmin);
  const height = Math.abs(ymax - ymin);
  const depth = Math.abs(zmax - zmin);

  if (width < 0.01 || height < 0.01 || depth < 0.01) return null;

  const center = new THREE.Vector3(
    xmin + width / 2, ymin + height / 2, zmin + depth / 2
  );

  return (
    <group position={center}>
      <mesh onClick={(e) => { e.stopPropagation(); onSelect(item.guid); }}>
        <boxGeometry args={[width, height, depth]} />
        <meshBasicMaterial visible={false} />
      </mesh>
      {isSelected && (
        <mesh>
          <boxGeometry args={[width + 0.05, height + 0.05, depth + 0.05]} />
          <meshBasicMaterial color="#00ffff" wireframe transparent opacity={0.8} />
        </mesh>
      )}
    </group>
  );
};

// Índices das 12 arestas que conectam os 8 cantos do OBB
// (ordem: 0-3 face inferior CCW, 4-7 face superior CCW)
const OBB_EDGES = new Uint16Array([
  0, 1, 1, 2, 2, 3, 3, 0,    // face inferior
  4, 5, 5, 6, 6, 7, 7, 4,    // face superior
  0, 4, 1, 5, 2, 6, 3, 7,    // arestas verticais
]);

// Desenha OBB wireframe a partir de 8 cantos [x,y,z] em coords Three.js
const ObbWireframe: React.FC<{ corners: number[][]; color: string; opacity?: number }> = ({
  corners, color, opacity = 0.4,
}) => {
  const geometry = React.useMemo(() => {
    const positions = new Float32Array(8 * 3);
    for (let i = 0; i < 8; i++) {
      positions[i * 3]     = corners[i][0];
      positions[i * 3 + 1] = corners[i][1];
      positions[i * 3 + 2] = corners[i][2];
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    g.setIndex(new THREE.BufferAttribute(OBB_EDGES, 1));
    return g;
  }, [corners]);

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color={color} transparent opacity={opacity} />
    </lineSegments>
  );
};

const BimBoundingBox: React.FC<{ item: any }> = ({ item }) => {
  // Preferência: OBB (8 cantos orientados) → fallback AABB
  const corners = item.obb_corners;
  const color = item.status?.cor || '#cccccc';

  if (corners && Array.isArray(corners) && corners.length === 8) {
    return <ObbWireframe corners={corners} color={color} opacity={0.4} />;
  }

  // ----- Fallback AABB -----
  const bbox = item.bbox_normalized || item.bbox;
  if (!bbox) {
    console.warn("Item sem bbox:", item.nome || item.guid);
    return null;
  }
  const { xmin, xmax, ymin, ymax, zmin, zmax } = bbox;
  const width = Math.abs(xmax - xmin);
  const height = Math.abs(ymax - ymin);
  const depth = Math.abs(zmax - zmin);
  if (width < 0.01 || height < 0.01 || depth < 0.01) return null;
  const center = new THREE.Vector3(
    xmin + width / 2, ymin + height / 2, zmin + depth / 2
  );
  return (
    <group position={center}>
      <mesh>
        <boxGeometry args={[width, height, depth]} />
        <meshBasicMaterial color={color} wireframe transparent opacity={0.4} />
      </mesh>
    </group>
  );
};

// --- COMPONENTE PRINCIPAL ---
export const DataView: React.FC<{
  result: AnalysisResult | null;
  resultAI?: AnalysisResult | null;
  analysisMode?: string;
}> = ({ result, resultAI, analysisMode = 'ml_v1' }) => {

  const displayResult = result || resultAI;
  if (!displayResult) {
    return <div className="p-10 text-slate-500">Aguardando dados...</div>;
  }

  // Suporta ambos os formatos: novo (resultados/estatisticas) e antigo (items/summary)
  const resultados = displayResult.resultados || displayResult.items || [];
  const estatisticas = displayResult.estatisticas || {
    total: displayResult.summary?.total_elements || 0,
    completos: 0,
    parciais: 0,
    iniciados: 0,
    ausentes: 0,
    progresso_geral: displayResult.summary?.progress_percentage || 0
  };

  if (resultados.length === 0) {
    return <div className="p-10 text-slate-500">Nenhum objeto encontrado.</div>;
  }

  // --- ESTADO DE SELEÇÃO ---
  const [selectedGuid, setSelectedGuid] = useState<string | null>(null);
  const tableRowRefs = useRef<Record<string, HTMLTableRowElement | null>>({});

  // Seleção no 3D só marca a linha (não rola a tabela automaticamente)

  // --- ESTADO DE VISIBILIDADE ---
  const [hiddenCategories, setHiddenCategories] = useState<Set<string>>(new Set());
  const [hiddenItems, setHiddenItems] = useState<Set<string>>(new Set());
  const [forcedVisibleItems, setForcedVisibleItems] = useState<Set<string>>(new Set());

  // Verifica se um item deve ser visível
  const isItemVisible = (item: any) => {
    const status = item.status?.code;
    const guid = item.guid;

    // Se forçado visível, sempre mostra
    if (forcedVisibleItems.has(guid)) return true;

    // Se categoria oculta, não mostra
    if (hiddenCategories.has(status)) return false;

    // Se item individualmente oculto, não mostra
    if (hiddenItems.has(guid)) return false;

    return true;
  };

  // Toggle de categoria inteira
  const toggleCategory = (statusCode: string) => {
    setHiddenCategories(prev => {
      const next = new Set(prev);
      if (next.has(statusCode)) {
        next.delete(statusCode);
        // Limpa forçados dessa categoria
        setForcedVisibleItems(f => {
          const newF = new Set(f);
          resultados.filter(i => i.status?.code === statusCode).forEach(i => newF.delete(i.guid));
          return newF;
        });
      } else {
        next.add(statusCode);
      }
      return next;
    });
  };

  // Toggle de item individual
  const toggleItemVisibility = (guid: string, statusCode: string) => {
    // Se categoria está oculta, força o item a aparecer
    if (hiddenCategories.has(statusCode)) {
      setForcedVisibleItems(prev => {
        const next = new Set(prev);
        if (next.has(guid)) {
          next.delete(guid);
        } else {
          next.add(guid);
        }
        return next;
      });
    } else {
      // Categoria visível: toggle normal de ocultar
      setHiddenItems(prev => {
        const next = new Set(prev);
        if (next.has(guid)) {
          next.delete(guid);
        } else {
          next.add(guid);
        }
        return next;
      });
    }
  };

  // Agrupa por tipo para o gráfico
  const statusByType = resultados.reduce((acc: any, item) => {
    if (!acc[item.tipo]) acc[item.tipo] = { total: 0, completo: 0, parcial: 0, detectado: 0 };
    acc[item.tipo].total++;
    if (item.status.code === 'COMPLETO') acc[item.tipo].completo++;
    if (item.status.code === 'PARCIAL') acc[item.tipo].parcial++;
    if (item.status.code === 'DETECTADO') acc[item.tipo].detectado++;
    return acc;
  }, {});

  const chartData = Object.entries(statusByType).map(([tipo, stats]: any) => ({
    name: tipo.replace('Ifc', ''),
    Total: stats.total,
    Completo: stats.completo,
    Parcial: stats.parcial,
    Detectado: stats.detectado || 0,
  }));

  // IMPORTANT: Removed h-full and overflow-y-auto from here to allow the parent container in App.tsx to handle scrolling.
  // This ensures the table at the bottom is accessible via the main page scrollbar.
  return (
    <div className="flex flex-col w-full p-6 gap-6 bg-slate-50">

      {/* Título */}
      <div className="shrink-0">
        <h2 className="text-2xl font-bold text-slate-800 flex items-center">
          <Box className="w-6 h-6 mr-2 text-indigo-600" />
          Digital Twin & KPIs
        </h2>
      </div>

      <div className="flex gap-6 min-h-[500px]"> {/* Altura mínima, permite scroll */}

        {/* --- ÁREA 3D --- */}
        <div className="flex-[3] bg-slate-900 rounded-xl overflow-hidden relative border border-slate-700">
          <Canvas
            camera={{ position: [5, 5, 5], fov: 50 }}
            gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
            dpr={[1, 2]}
          >
            {/* LUZES */}
            <ambientLight intensity={0.8} />
            <pointLight position={[10, 10, 10]} />

            {/* GUIAS VISUAIS */}
            <Grid args={[20, 20]} cellColor="white" sectionColor="red" />
            <primitive object={new THREE.AxesHelper(5)} /> {/* Eixos X,Y,Z coloridos */}

            {/* FERRAMENTA DE MEDIÇÃO (inclui OrbitControls) */}
            <MeasureTool />

            {/* NUVEM GLOBAL (fundo cinza, mostra forma completa do scan) */}
            <GlobalPointsCloud jsonFile={(displayResult as any)?.global_cloud} />

            {/* OBJETOS BIM: BOUNDING BOXES + PONTOS (filtrado por visibilidade) */}
            {resultados.filter(item => isItemVisible(item)).map((item, i) => (
              <group key={item.guid || i}>
                {/* Hitbox clicável para seleção */}
                <SelectableHitbox
                  item={item}
                  isSelected={selectedGuid === item.guid}
                  onSelect={(guid) => setSelectedGuid(prev => prev === guid ? null : guid)}
                />

                <BimBoundingBox item={item} />

                {/* Nuvem de pontos da execução (se existir) */}
                {item.json_file && (
                  <ObjectPoints
                    jsonFile={item.json_file}
                    color={selectedGuid === item.guid ? '#00ffff' : (item.status.cor || '#00ff00')}
                  />
                )}
              </group>
            ))}
          </Canvas>
        </div>

        {/* --- GRÁFICOS (DIREITA) --- */}
        <div className="flex-[2] flex flex-col gap-4 overflow-y-auto">
          {/* Cards */}
          <div className="grid grid-cols-2 gap-4">
              <div className="bg-white p-4 rounded shadow border border-slate-200">
                <div className="text-xs text-slate-500">Progresso</div>
                <div className="text-2xl font-bold">{estatisticas.progresso_geral.toFixed(0)}%</div>
              </div>
              <div className="bg-white p-4 rounded shadow border border-slate-200">
                <div className="text-xs text-slate-500">Elementos</div>
                <div className="text-2xl font-bold">{estatisticas.completos + estatisticas.parciais}/{estatisticas.total}</div>
              </div>
          </div>

          {/* Gráfico */}
          <div className="bg-white p-4 rounded shadow border border-slate-200 h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="Total" fill="#cbd5e1" />
                <Bar dataKey="Completo" fill="#4caf50" />
                <Bar dataKey="Parcial" fill="#ff9800" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* TABELA DE OBJETOS */}
      <div className="mt-6 bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex justify-between items-center">
          <h3 className="font-semibold text-slate-700">📋 Lista de Objetos ({resultados.length} itens)</h3>
          <div className="flex gap-2 text-xs">
                <span
                  onClick={() => toggleCategory('COMPLETO')}
                  className={`px-2 py-1 bg-green-100 text-green-700 rounded cursor-pointer hover:ring-2 flex items-center gap-1 ${hiddenCategories.has('COMPLETO') ? 'opacity-50' : ''}`}
                >
                  ✅ Completo {hiddenCategories.has('COMPLETO') ? '👁️‍🗨️' : '👁️'}
                </span>
                <span
                  onClick={() => toggleCategory('PARCIAL')}
                  className={`px-2 py-1 bg-orange-100 text-orange-700 rounded cursor-pointer hover:ring-2 flex items-center gap-1 ${hiddenCategories.has('PARCIAL') ? 'opacity-50' : ''}`}
                >
                  ⚠️ Parcial {hiddenCategories.has('PARCIAL') ? '👁️‍🗨️' : '👁️'}
                </span>
                <span
                  onClick={() => toggleCategory('INICIADO')}
                  className={`px-2 py-1 bg-blue-100 text-blue-700 rounded cursor-pointer hover:ring-2 flex items-center gap-1 ${hiddenCategories.has('INICIADO') ? 'opacity-50' : ''}`}
                >
                  🔶 Iniciado {hiddenCategories.has('INICIADO') ? '👁️‍🗨️' : '👁️'}
                </span>
                <span
                  onClick={() => toggleCategory('AUSENTE')}
                  className={`px-2 py-1 bg-red-100 text-red-700 rounded cursor-pointer hover:ring-2 flex items-center gap-1 ${hiddenCategories.has('AUSENTE') ? 'opacity-50' : ''}`}
                >
                  ❌ Ausente {hiddenCategories.has('AUSENTE') ? '👁️‍🗨️' : '👁️'}
                </span>
                <span
                  onClick={() => toggleCategory('ADICAO')}
                  className={`px-2 py-1 bg-sky-100 text-sky-700 rounded cursor-pointer hover:ring-2 flex items-center gap-1 ${hiddenCategories.has('ADICAO') ? 'opacity-50' : ''}`}
                  title="Construído fora do plano (detectado no scan, ausente no IFC)"
                >
                  ➕ Adição {hiddenCategories.has('ADICAO') ? '👁️‍🗨️' : '👁️'}
                </span>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-100 sticky top-0">
              <tr>
                <th className="text-left px-4 py-2 font-medium text-slate-600">Nome</th>
                <th className="text-left px-4 py-2 font-medium text-slate-600">Tipo</th>
                <th className="text-center px-4 py-2 font-medium text-slate-600">Status</th>
                <th className="text-center px-2 py-2 font-medium text-slate-600" title="Visibilidade">👁️</th>
                <th className="text-center px-4 py-2 font-medium text-slate-600">Cobertura</th>
                <th className="text-center px-4 py-2 font-medium text-slate-600">Altura (Exec/Plan)</th>
              </tr>
            </thead>
            <tbody>
              {resultados.map((item: any, idx: number) => (
                <tr
                  key={item.guid || idx}
                  ref={(el) => { tableRowRefs.current[item.guid] = el; }}
                  onClick={() => setSelectedGuid(prev => prev === item.guid ? null : item.guid)}
                  className={`border-b border-slate-100 cursor-pointer transition-colors ${
                    selectedGuid === item.guid ? 'bg-cyan-100 ring-2 ring-cyan-400' :
                    item.status.code === 'AUSENTE' ? 'bg-red-50 hover:bg-slate-50' :
                    item.status.code === 'PARCIAL' ? 'bg-orange-50 hover:bg-slate-50' :
                    item.status.code === 'ADICAO'  ? 'bg-sky-50 hover:bg-slate-50' :
                    'hover:bg-slate-50'
                  }`}>
                  <td className="px-4 py-2 font-medium text-slate-800">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span>{item.nome}</span>
                      {item.cross_floor && (
                        <span
                          className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-amber-100 text-amber-800 border border-amber-300"
                          title={
                            `Estrutura vertical atravessando andares\n` +
                            `Altura total: ${item.altura_total?.toFixed(2)}m ` +
                            `(Z ${item.z_total?.[0]?.toFixed(2)} → ${item.z_total?.[1]?.toFixed(2)})\n` +
                            `Atravessa: ${(item.andares_atravessa || []).join(', ')}\n` +
                            `Storey IFC: ${item.storey_original_ifc}\n` +
                            (item.fatia_atual
                              ? `Fatia aqui: ${item.fatia_atual.altura?.toFixed(2)}m ` +
                                `(Z ${item.fatia_atual.z_range?.[0]?.toFixed(2)} → ${item.fatia_atual.z_range?.[1]?.toFixed(2)})`
                              : '')
                          }
                        >
                          ⚡ Cross-floor
                        </span>
                      )}
                    </div>
                    {item.cross_floor && (
                      <div className="text-[11px] text-slate-500 mt-0.5">
                        Total {item.altura_total?.toFixed(2)}m · atravessa{' '}
                        {(item.andares_atravessa || []).length} andar
                        {(item.andares_atravessa || []).length === 1 ? '' : 'es'}
                        {item.fatia_atual && ` · fatia ${item.fatia_atual.altura?.toFixed(2)}m`}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-2 text-slate-600">{item.tipo.replace('Ifc', '')}</td>
                  <td className="px-4 py-2 text-center">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      item.status.code === 'DETECTADO' ? 'bg-cyan-100 text-cyan-700' :
                      item.status.code === 'COMPLETO' ? 'bg-green-100 text-green-700' :
                      item.status.code === 'PARCIAL' ? 'bg-orange-100 text-orange-700' :
                      item.status.code === 'INICIADO' ? 'bg-blue-100 text-blue-700' :
                      item.status.code === 'ADICAO'   ? 'bg-sky-100 text-sky-700' :
                      'bg-red-100 text-red-700'
                      }`}>
                      {item.status.emoji} {item.status.texto}
                    </span>
                  </td>
                  <td
                    className="px-2 py-2 text-center cursor-pointer hover:bg-slate-100 text-lg"
                    onClick={() => toggleItemVisibility(item.guid, item.status.code)}
                    title={isItemVisible(item) ? 'Clique para ocultar' : 'Clique para mostrar'}
                  >
                    {isItemVisible(item) ? '👁️' : '👁️‍🗨️'}
                  </td>
                  <td className="px-4 py-2 text-center text-slate-600">
                    {item.cobertura_vertical !== undefined ? `${(item.cobertura_vertical * 100).toFixed(0)}%` :
                      item.cobertura !== undefined ? `${item.cobertura}%` : '-'}
                  </td>
                  <td className="px-4 py-2 text-center text-slate-600">
                    {item.dimensoes ? `${item.dimensoes.executado?.z?.toFixed(1) || 0}m / ${item.dimensoes.planejado?.z?.toFixed(1) || 0}m` : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};