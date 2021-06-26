#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define BIG_NUMBER 1e10
#define SMALL_NUMBER 1e-5
static std::default_random_engine generator;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	if(is_initialized) return;
	num_particles = 90;
	std::normal_distribution<double> x_dist(x, std[0]);
	std::normal_distribution<double> y_dist(y, std[1]);
	std::normal_distribution<double> theta_dist(theta, std[2]);
	for(int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = x_dist(generator);
		particle.y = y_dist(generator);
		particle.theta = theta_dist(generator);
		particle.weight = 1.0;
		particles.push_back(particle);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
  double velocity, double yaw_rate) {
	std::normal_distribution<double> x_dist(0, std_pos[0]);
	std::normal_distribution<double> y_dist(0, std_pos[1]);
	std::normal_distribution<double> theta_dist(0, std_pos[2]);
	for(int i = 0; i < num_particles; i++) {
    double theta = particles[i].theta;
		if(fabs(yaw_rate) < SMALL_NUMBER) {  
			particles[i].x += delta_t * velocity * cos(theta);
			particles[i].y += delta_t * velocity * sin(theta);
		} else {
      double delta_theta = delta_t * yaw_rate;
			particles[i].x += velocity * (sin(theta + delta_theta) - sin(theta)) / yaw_rate;
			particles[i].y -= velocity * (cos(theta + delta_theta) - cos(theta)) / yaw_rate;
			particles[i].theta += delta_theta;
		}
		particles[i].x += x_dist(generator);
		particles[i].y += y_dist(generator);
		particles[i].theta += theta_dist(generator);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, 
  std::vector<LandmarkObs>& observations) {}

inline double square(double x) {
  return x * x;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  double squared_sensor_range = square(sensor_range);
  double sx = std_landmark[0], sy = std_landmark[1];
	for(int i = 0; i < num_particles; i++) {
    auto particle = particles[i];
		std::vector<LandmarkObs> close_landmarks;
		for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      auto landmark = map_landmarks.landmark_list[j];
      double dx = landmark.x_f - particle.x;
      double dy = landmark.y_f - particle.y;
      if(square(dx) + square(dy) < squared_sensor_range) 
				close_landmarks.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
		}
		particles[i].weight = 1.0;
    for(unsigned int j = 0; j < observations.size(); j++) {
      auto observation = observations[j];
      double ct = cos(particle.theta), st = sin(particle.theta);
      double ox = ct * observation.x - st * observation.y + particle.x;
			double oy = st * observation.x + ct * observation.y + particle.y;
			double lx, ly, min_distance_2 = BIG_NUMBER;
			for(unsigned int k = 0; k < close_landmarks.size(); k++) {
        auto close_landmark = close_landmarks[k];
				double dx = ox - close_landmark.x;
				double dy = oy - close_landmark.y;
				double distance_2 = square(dx) + square(dy);
				if(distance_2 < min_distance_2) {
					min_distance_2 = distance_2;
					lx = close_landmarks[k].x;
					ly = close_landmarks[k].y;
				}
			}
      double power = square((lx - ox) / sx) + square((ly - oy) / sy);
			double gaussian = exp(-0.5 * power) / (2 * M_PI * sx * sy);
			particles[i].weight *= gaussian;
		}
	}
}

void ParticleFilter::resample() {
	double max_weight = 0.0;
	for(int i = 0; i < num_particles; i++)
		if(particles[i].weight > max_weight)
			max_weight = particles[i].weight;
	std::uniform_real_distribution<double> wheel_dist(0.0, max_weight);
	std::uniform_int_distribution<int> index_dist(0, num_particles - 1);
	int index = index_dist(generator);
	double wheel = 0.0;
	std::vector<Particle> new_particles;
	for(int i = 0; i < num_particles; i++) {
		wheel += 2.0 * wheel_dist(generator);
		while(wheel > particles[index].weight) {
			wheel -= particles[index].weight;
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}
	particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
  const std::vector<int>& associations,
  const std::vector<double>& sense_x, const std::vector<double>& sense_y) {
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

std::string ParticleFilter::getAssociations(Particle best) {
  std::vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);
  return s;
}

std::string ParticleFilter::getSenseCoord(Particle best, std::string coord) {
  std::vector<double> v = coord == "X" ? best.sense_x : best.sense_y;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);
  return s;
}