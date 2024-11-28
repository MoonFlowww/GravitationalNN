#include "raylib.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>

const float G = 500.0f;
const float MIN_DISTANCE = 10.0f;
const double LEARNING_RATE = 0.1;
const double BETA1 = 0.9;
const double BETA2 = 0.999;
const double EPSILON = 1e-3;

struct Object {
    Vector2 position;
    Vector2 velocity;
    float mass;  // w
    float bias;  //b
    Color color;
    float m;
    float v;
};

Vector2 CalculateGravitationalForce(const Object& obj1, const Object& obj2) {
    Vector2 direction = { obj2.position.x - obj1.position.x, obj2.position.y - obj1.position.y };
    float distance = sqrt(direction.x * direction.x + direction.y * direction.y);

    if (distance < MIN_DISTANCE) {
        distance = MIN_DISTANCE;
    }

    direction.x /= distance;
    direction.y /= distance;

    float forceMagnitude = G * (obj1.mass * obj2.mass) / (distance * distance);
    Vector2 force = { forceMagnitude * direction.x, forceMagnitude * direction.y };

    return force;
}

const int screenWidth = 1200;
const int screenHeight = 700;
const int spacingX = 200;
const int hiddenLayerCount = 2;


void InitializeSimulation() {
    InitWindow(screenWidth, screenHeight, "Simulation de l'Apprentissage Machine - Raylib");
    SetTargetFPS(60);
}

void UpdateSimulation(std::vector<Object>& asteroids, std::vector<Vector2>& targetPositions, std::vector<std::vector<Object>>& layers, std::vector<bool>& asteroidsStopped, std::vector<float>& errors, bool& paused);
void UpdateMassesRL(std::vector<std::vector<Object>>& layers, const std::vector<Object>& asteroids, const std::vector<Vector2>& targetPositions, const std::vector<float>& errors, int episode);

void DrawSimulation(const std::vector<Object>& asteroids, const std::vector<std::vector<Object>>& layers, const std::vector<Vector2>& targetPositions, const std::vector<bool>& asteroidsStopped, const std::vector<float>& errors, float improvement);

int main() {
    InitializeSimulation();

    const int inputLayerNeurons = 2;
    const int neuronsPerHiddenLayer = 4;
    const int outputLayerNeurons = 1;
    const int numAsteroids = 3;

    const int spacingY = 100;

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    std::vector<std::vector<Object>> layers;
    float inputX = screenWidth / 2 - (spacingX * (hiddenLayerCount + 1)) / 2;

    std::vector<Object> inputLayer;
    for (int i = 0; i < inputLayerNeurons; ++i) {
        float inputY = screenHeight / 2 - (inputLayerNeurons * spacingY) / 2 + i * spacingY;
        inputLayer.push_back(Object{
            {inputX, inputY},
            {0.0f, 0.0f},
            static_cast<float>(1 + rand() % 20),
            static_cast<float>(rand() % 50 - 25),
            BLUE,
            0.0f,  // m for Adam
            0.0f   // v for Adam
            });
    }
    layers.push_back(inputLayer);

    for (int i = 0; i < hiddenLayerCount; ++i) {
        std::vector<Object> hiddenLayer;
        float x = screenWidth / 2 - (spacingX * (hiddenLayerCount + 1)) / 2 + (i + 1) * spacingX;
        for (int j = 0; j < neuronsPerHiddenLayer; ++j) {
            float y = screenHeight / 2 - (neuronsPerHiddenLayer * spacingY) / 2 + j * spacingY;
            hiddenLayer.push_back(Object{
                {x, y},
                {0.0f, 0.0f},
                static_cast<float>(1 + rand() % 20),
                static_cast<float>(rand() % 50 - 25),
                BLUE,
                0.0f,  // m for Adam
                0.0f   // v for Adam
                });
        }
        layers.push_back(hiddenLayer);
    }

    std::vector<Object> outputLayer;
    outputLayer.push_back(Object{
        {screenWidth / 2 + (spacingX * hiddenLayerCount) / 2, screenHeight / 2},
        {0.0f, 0.0f},
        static_cast<float>(1 + rand() % 20),
        static_cast<float>(rand() % 50 - 25),
        BLUE,
        0.0f,  // m for Adam
        0.0f   // v for Adam
        });
    layers.push_back(outputLayer);

    std::vector<Object> asteroids;
    std::vector<Color> colors = { RED, GREEN, VIOLET };
    std::vector<Vector2> targetPositions;
    for (int i = 0; i < numAsteroids; ++i) {
        float startY = screenHeight / 2 - 200 + i * 200;  // star position inputs
        asteroids.push_back(Object{ {inputX - spacingX, startY}, {1.0f, 0}, 1, 0, colors[i], 0.0f, 0.0f });

        float targetY = screenHeight / 2 - 200 + i * 200;
        targetPositions.push_back(Vector2{ static_cast<float>(screenWidth), targetY }); // shift
    }

    std::vector<bool> asteroidsStopped(numAsteroids, false);
    std::vector<float> errors(numAsteroids, 0.0f);
    float prevError = 0.0f, improvement = 0.0f;
    bool paused = false;

    for (int episode = 0; episode < 1000; ++episode) {
        for (int i = 0; i < numAsteroids; ++i) {
            asteroids[i].position = { static_cast<float>(inputX - spacingX), static_cast<float>(screenHeight / 2 - 200 + i * 200) };
            asteroids[i].velocity = { 1.0f, 0 };
            asteroidsStopped[i] = false;
            errors[i] = 0.0f;
        }

        while (!WindowShouldClose()) {
            UpdateSimulation(asteroids, targetPositions, layers, asteroidsStopped, errors, paused);
            if (!paused) {
                bool allStopped = true;
                for (bool stopped : asteroidsStopped) {
                    allStopped &= stopped;
                }

                if (allStopped) {
                    UpdateMassesRL(layers, asteroids, targetPositions, errors, episode);
                    float currentError = 0.0f;
                    for (float error : errors) {
                        currentError += error;
                    }
                    if (episode > 0) {
                        improvement = 100.0f * (prevError - currentError) / prevError;
                    }
                    prevError = currentError;
                    break;
                }
            }
            DrawSimulation(asteroids, layers, targetPositions, asteroidsStopped, errors, improvement);
        }

        std::cout << "Episode " << episode + 1 << ", Improvement: " << improvement << "%" << std::endl;
    }

    CloseWindow();
    return 0;
}

void UpdateSimulation(std::vector<Object>& asteroids, std::vector<Vector2>& targetPositions, std::vector<std::vector<Object>>& layers, std::vector<bool>& asteroidsStopped, std::vector<float>& errors, bool& paused) {
    if (IsKeyPressed(KEY_R)) {
        for (auto& asteroid : asteroids) {
            asteroid.position = { 100, 600 };
            asteroid.velocity = { 1.0f, 0 };
        }
        std::fill(asteroidsStopped.begin(), asteroidsStopped.end(), false);
        std::fill(errors.begin(), errors.end(), 0.0f);
    }

    if (IsKeyPressed(KEY_SPACE)) {
        paused = !paused;
        if (paused) {
            SetTargetFPS(0);
        }
        else {
            SetTargetFPS(60);
        }
    }

    if (!paused) {
        for (size_t i = 0; i < asteroids.size(); ++i) {
            if (!asteroidsStopped[i]) {
                Vector2 totalForce = { 0, 0 };
                for (const auto& layer : layers) {
                    for (const auto& neuron : layer) {
                        float adjustedY = neuron.position.y + neuron.bias; 
                        Vector2 force = CalculateGravitationalForce(asteroids[i], neuron); 
                        totalForce.x += force.x;
                        totalForce.y += force.y;
                    }
                }

                asteroids[i].velocity.x += (totalForce.x / asteroids[i].mass);
                asteroids[i].velocity.y += (totalForce.y / asteroids[i].mass);

                asteroids[i].position.x += asteroids[i].velocity.x;
                asteroids[i].position.y += asteroids[i].velocity.y;

                errors[i] = pow(asteroids[i].position.x - targetPositions[i].x, 2) + pow(asteroids[i].position.y - targetPositions[i].y, 2);

                if (asteroids[i].position.x < 0 || asteroids[i].position.x > 1200 ||
                    asteroids[i].position.y < 0 || asteroids[i].position.y > 700 ||
                    asteroids[i].position.x >= targetPositions[i].x + spacingX) {
                    asteroidsStopped[i] = true;
                }
            }
        }
    }
}

void UpdateMassesRL(std::vector<std::vector<Object>>& layers, const std::vector<Object>& asteroids, const std::vector<Vector2>& targetPositions, const std::vector<float>& errors, int episode) {
    for (auto& layer : layers) {
        for (auto& neuron : layer) {
            for (size_t i = 0; i < asteroids.size(); ++i) {
                float reward = -errors[i];
                float gradient = reward * (1.0f / (MIN_DISTANCE + errors[i]));

                // Adam optimizer updates
                neuron.m = BETA1 * neuron.m + (1 - BETA1) * gradient;
                neuron.v = BETA2 * neuron.v + (1 - BETA2) * (gradient * gradient);

                float m_hat = neuron.m / (1 - pow(BETA1, episode + 1));
                float v_hat = neuron.v / (1 - pow(BETA2, episode + 1));

                neuron.mass += LEARNING_RATE * m_hat / (sqrt(v_hat) + EPSILON);
                neuron.bias += LEARNING_RATE * m_hat / (sqrt(v_hat) + EPSILON);

                if (neuron.mass < 1.0f) neuron.mass = 1.0f;
                if (neuron.mass > 100.0f) neuron.mass = 100.0f;
            }
        }
    }
}

void DrawSimulation(const std::vector<Object>& asteroids, const std::vector<std::vector<Object>>& layers, const std::vector<Vector2>& targetPositions, const std::vector<bool>& asteroidsStopped, const std::vector<float>& errors, float improvement) {
    BeginDrawing();
    ClearBackground(RAYWHITE);

    for (size_t i = 0; i < layers.size(); ++i) {
        for (const auto& neuron : layers[i]) {
            float radius = 8.0f + neuron.mass * 0.8f;
            float adjustedY = neuron.position.y + neuron.bias;  
            DrawCircleV({ neuron.position.x, adjustedY }, radius, neuron.color); 
        }
    }

    for (size_t i = 0; i < asteroids.size(); ++i) {
        DrawCircleV(asteroids[i].position, 10, Fade(asteroids[i].color, 0.8f));
    }

    for (size_t i = 0; i < targetPositions.size(); ++i) {
        DrawCircleV(targetPositions[i], 30, Fade(asteroids[i].color, 0.5f)); 
        DrawText("Target", targetPositions[i].x - 40, targetPositions[i].y - 10, 20, DARKGREEN);
    }

    DrawText(TextFormat("Improvement: %.2f%%", improvement), 50, 50, 20, BLACK);

    DrawText("Appuyez sur 'R' pour recommencer", 50, 80, 20, DARKGRAY);

    EndDrawing();
}
