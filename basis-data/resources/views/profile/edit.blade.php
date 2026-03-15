{{-- edit.blade.php --}}
<x-app-layout>
    <x-slot name="header">
        <div class="title-box mb-0">
            <h2 class="font-semibold text-xl text-white leading-tight d-flex align-items-center">
                <i class="fas fa-user me-2"></i>
                {{ __('Profile') }}
            </h2>
        </div>
    </x-slot>

    <div class="py-8">
        <div class="container mx-auto px-4">
            <!-- Profil Header dengan Foto dan Info -->
            <div class="profile-header-card mb-8 animate-fade-in">
                <div class="row align-items-center">
                    <div class="col-md-3 text-center mb-4 mb-md-0">
                        <div class="profile-avatar">
                            @if(auth()->user()->foto)
                                <img src="{{ asset('storage/users/' . auth()->user()->foto) }}" alt="{{ auth()->user()->name }}" class="avatar-img">
                            @else
                                <div class="avatar-placeholder">
                                    <i class="fas fa-user fs-1 text-light"></i>
                                </div>
                            @endif
                        </div>
                        @if(auth()->user()->peran == 'admin')
                            <span class="role-badge admin">Admin</span>
                        @elseif(auth()->user()->peran == 'petani')
                            <span class="role-badge petani">Petani</span>
                        @elseif(auth()->user()->peran == 'pengepul')
                            <span class="role-badge pengepul">Pengepul</span>
                        @elseif(auth()->user()->peran == 'distributor')
                            <span class="role-badge distributor">Distributor</span>
                        @endif
                    </div>
                    <div class="col-md-9">
                        <div class="profile-info">
                            <h3 class="text-white mb-2">{{ auth()->user()->name }}</h3>
                            <p class="text-light mb-1">Email: {{ auth()->user()->email }}</p>
                            <p class="text-light mb-2">Bergabung sejak: {{ auth()->user()->created_at->format('d M Y') }}</p>
                            @if(session('status') === 'profile-updated' || session('status') === 'password-updated')
                                <div class="alert-success-profile">
                                    <i class="fas fa-check-circle me-2"></i>
                                    {{ session('status') === 'profile-updated' ? 'Profil berhasil diperbarui.' : 'Password berhasil diperbarui.' }}
                                </div>
                            @endif
                        </div>
                    </div>
                </div>
            </div>

            <div class="row space-y-6">
                <!-- Update Profile Information -->
                <div class="col-12">
                    <div class="profile-section-card animate-slide-in">
                        <div class="section-header">
                            <h3 class="text-white mb-0">
                                <i class="fas fa-user-edit me-2"></i>{{ __('Profile Information') }}
                            </h3>
                        </div>
                        <div class="section-body">
                            <div class="max-w-xl mx-auto">
                                @include('profile.partials.update-profile-information-form')
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Update Password -->
                <div class="col-12">
                    <div class="profile-section-card animate-slide-in" style="animation-delay: 0.1s;">
                        <div class="section-header">
                            <h3 class="text-white mb-0">
                                <i class="fas fa-lock me-2"></i>{{ __('Update Password') }}
                            </h3>
                        </div>
                        <div class="section-body">
                            <div class="max-w-xl mx-auto">
                                @include('profile.partials.update-password-form')
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Delete Account -->
                <div class="col-12">
                    <div class="profile-section-card animate-slide-in" style="animation-delay: 0.2s;">
                        <div class="section-header">
                            <h3 class="text-white mb-0">
                                <i class="fas fa-exclamation-triangle me-2"></i>{{ __('Delete Account') }}
                            </h3>
                        </div>
                        <div class="section-body">
                            <div class="max-w-xl mx-auto">
                                @include('profile.partials.delete-user-form')
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</x-app-layout>